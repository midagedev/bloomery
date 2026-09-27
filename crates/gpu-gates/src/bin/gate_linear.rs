//! GPU gate for the linear-attention kernels (`bloomery_gpu::linear`), with
//! no model: synthetic layers and tokens at Qwen3.6-35B-A3B's shapes (16
//! query/key heads, 32 value heads, 128 wide, 8,192 conv channels, the
//! tiled key-head map), from fixed seeds. Each kernel's output is held to its
//! host rule bit for bit; every clause draws its inputs from the host, not
//! from another kernel, so a defect turns only its own kernel's lines red.
//!
//! Clauses:
//! 1. `math`: [`expf_ik`] and [`logf_poly`] within [`MATH_ULPS`] of the f64
//!    functions (exp on [−87, 88], log on [1, 1 + e²⁰], the range softplus
//!    passes it).
//! 2. `conv`: the conv and prep of 512, 2 and 1 tokens from a random
//!    incoming conv state — output, β, decay and the new state — and 512
//!    one-token launches chained through the state equal to the 512-token
//!    launch.
//! 3. `delta`: a 512-token prompt from a zero state (`o` and the state after
//!    it), one decode step from that state, a one-token prompt from a zero
//!    state, the grouped key-head map on 8 tokens, and 512 one-token
//!    launches chained through the state equal to the 512-token launch.
//! 4. `norm_gate`: 512 and 1 tokens.
//! 5. `graph`: one decode token's conv, delta and norm captured as a graph:
//!    three nodes, the replay equal to the eager launches.
//! 6. `depth`: 4,096 tokens through the conv and the delta step on the card
//!    in launches of 6, 1,018 and 3,072 tokens, the state after the last
//!    equal to the host rule over all 4,096 at once; then (reported, not
//!    judged) how far the f32 state at depths 6, 1,024 and 4,096 sits from
//!    an f64 run of the same recurrence on the same inputs.
//! 7. `fault`: a NaN input and an overflow from finite inputs for each
//!    kernel, each launched with a layer's sink: the word names that layer
//!    and the kernel's site, the values the planted input reaches are not
//!    finite, every other value is the clean launch's bit for bit, and the
//!    word is clean before and after.
//! 8. `shape`: the three entries compile with no local depot.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_linear: built without the `gpu` feature; see `just gate-gpu-linear`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_linear", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::linear::conv::{ConvArgs, ConvOut, conv_prep_host};
    use bloomery_gpu::linear::delta::{DeltaArgs, DeltaOut, delta_host};
    use bloomery_gpu::linear::norm_gate::{NormGateArgs, norm_gate_host};
    use bloomery_gpu::linear::{
        CONV_ROWS, CONV_TAPS, GATE_SILU, HEAD, KHeadMap, LinearKernels, LinearShape, expf_ik,
        logf_poly,
    };
    use bloomery_gpu::{Fault, FaultSink, FaultSite, Gpu};
    use bloomery_gpu_gates::{
        GateError, bits_equal, checks_failed, max_ulps, no_local_depot, same_bits, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer};

    /// Qwen3.6-35B-A3B's linear layers.
    const SHAPE: LinearShape = LinearShape {
        n_k: 16,
        n_v: 32,
        map: KHeadMap::Tiled,
    };
    /// The file's `layer_norm_rms_epsilon`, which both norms use.
    const EPS: f32 = 1e-6;
    /// The prompt length of the prefill clauses: one ubatch.
    const PROMPT: usize = 512;
    /// The depths of the drift report, and the launch lengths that reach them.
    const DEPTHS: [usize; 3] = [6, 1024, 4096];
    /// Band for [`expf_ik`] and [`logf_poly`] against the f64 functions
    /// rounded to f32, in ulps: ik's `v_expf` is ARM's `v_expf` polynomial
    /// (documented 1.45 ulp), Cephes `logf` a relative 7.6e-8 (under one ulp)
    /// before the final two fused steps round.
    const MATH_ULPS: u32 = 2;

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

    /// One layer's parameters.
    struct Layer {
        w: Vec<f32>,
        dt: Vec<f32>,
        sa: Vec<f32>,
        gain: Vec<f32>,
    }

    impl Layer {
        /// Conv taps in [−0.5, 0.5); `ssm_a = −e^u`, `u` uniform over the
        /// file's range (−72.33 … −0.0186); `dt_bias` in [−1, 1); the norm
        /// gain in [0.5, 1) (the file's 0.52 … 0.96).
        fn new(seed: u64) -> Layer {
            let mut r = Lcg(seed);
            let ch = SHAPE.channels();
            let (lo, hi) = (0.0186f32.ln(), 72.33f32.ln());
            Layer {
                w: r.fill(CONV_TAPS * ch, -0.5, 0.5),
                dt: r.fill(SHAPE.n_v, -1.0, 1.0),
                sa: (0..SHAPE.n_v).map(|_| -r.uniform(lo, hi).exp()).collect(),
                gain: r.fill(HEAD, 0.5, 1.0),
            }
        }
    }

    /// `m` tokens of the projections the kernels read.
    struct Tokens {
        x: Vec<f32>,
        b: Vec<f32>,
        a: Vec<f32>,
    }

    impl Tokens {
        fn new(m: usize, seed: u64) -> Tokens {
            let mut r = Lcg(seed);
            Tokens {
                x: r.fill(m * SHAPE.channels(), -1.0, 1.0),
                b: r.fill(m * SHAPE.n_v, -3.0, 3.0),
                a: r.fill(m * SHAPE.n_v, -3.0, 3.0),
            }
        }

        /// Tokens `t0 .. t0 + m`.
        fn slice(&self, t0: usize, m: usize) -> Tokens {
            let (ch, nv) = (SHAPE.channels(), SHAPE.n_v);
            Tokens {
                x: self.x[t0 * ch..(t0 + m) * ch].to_vec(),
                b: self.b[t0 * nv..(t0 + m) * nv].to_vec(),
                a: self.a[t0 * nv..(t0 + m) * nv].to_vec(),
            }
        }
    }

    /// The delta step's inputs for `m` tokens: the conv's outputs.
    struct Step {
        qkv: Vec<f32>,
        beta: Vec<f32>,
        decay: Vec<f32>,
    }

    impl Step {
        fn of(c: &ConvOut) -> Step {
            Step {
                qkv: c.y.clone(),
                beta: c.beta.clone(),
                decay: c.decay.clone(),
            }
        }

        fn slice(&self, t0: usize, m: usize) -> Step {
            let (ch, nv) = (SHAPE.channels(), SHAPE.n_v);
            Step {
                qkv: self.qkv[t0 * ch..(t0 + m) * ch].to_vec(),
                beta: self.beta[t0 * nv..(t0 + m) * nv].to_vec(),
                decay: self.decay[t0 * nv..(t0 + m) * nv].to_vec(),
            }
        }
    }

    struct Ctx<'a> {
        gpu: &'a Gpu,
        k: &'a LinearKernels,
    }

    impl Ctx<'_> {
        fn stream(&self) -> &CudaStream {
            self.gpu.stream()
        }

        fn conv(
            &self,
            fault: FaultSink,
            l: &Layer,
            tk: &Tokens,
            state_in: &[f32],
            m: usize,
        ) -> Result<ConvOut, GateError> {
            let s = self.stream();
            let (ch, nv) = (SHAPE.channels(), SHAPE.n_v);
            let x = DeviceBuffer::from_host(s, &tk.x)?;
            let b = DeviceBuffer::from_host(s, &tk.b)?;
            let a = DeviceBuffer::from_host(s, &tk.a)?;
            let w = DeviceBuffer::from_host(s, &l.w)?;
            let dt = DeviceBuffer::from_host(s, &l.dt)?;
            let sa = DeviceBuffer::from_host(s, &l.sa)?;
            let si = DeviceBuffer::from_host(s, state_in)?;
            let mut y = DeviceBuffer::from_host(s, &vec![0.0f32; m * ch])?;
            let mut beta = DeviceBuffer::from_host(s, &vec![0.0f32; m * nv])?;
            let mut decay = DeviceBuffer::from_host(s, &vec![0.0f32; m * nv])?;
            let mut so = DeviceBuffer::from_host(s, &vec![0.0f32; CONV_ROWS * ch])?;
            self.k.conv.enqueue_conv_prep(
                s,
                ConvArgs {
                    x: &x,
                    b_raw: &b,
                    a_raw: &a,
                    w: &w,
                    dt_bias: &dt,
                    ssm_a: &sa,
                    state_in: &si,
                    shape: SHAPE,
                    eps: EPS,
                    m,
                    fault,
                    y: &mut y,
                    beta: &mut beta,
                    decay: &mut decay,
                    state_out: &mut so,
                },
            )?;
            s.synchronize()?;
            Ok(ConvOut {
                y: y.to_host_vec(s)?,
                beta: beta.to_host_vec(s)?,
                decay: decay.to_host_vec(s)?,
                state: so.to_host_vec(s)?,
            })
        }

        fn delta(
            &self,
            fault: FaultSink,
            st: &Step,
            state_in: &[f32],
            shape: LinearShape,
            m: usize,
        ) -> Result<DeltaOut, GateError> {
            let s = self.stream();
            let qkv = DeviceBuffer::from_host(s, &st.qkv)?;
            let beta = DeviceBuffer::from_host(s, &st.beta)?;
            let decay = DeviceBuffer::from_host(s, &st.decay)?;
            let si = DeviceBuffer::from_host(s, state_in)?;
            let mut o = DeviceBuffer::from_host(s, &vec![0.0f32; m * shape.n_v * HEAD])?;
            let mut so = DeviceBuffer::from_host(s, &vec![0.0f32; shape.state_len()])?;
            self.k.delta.enqueue_delta(
                s,
                DeltaArgs {
                    qkv: &qkv,
                    beta: &beta,
                    decay: &decay,
                    state_in: &si,
                    shape,
                    m,
                    fault,
                    o: &mut o,
                    state_out: &mut so,
                },
            )?;
            s.synchronize()?;
            Ok(DeltaOut {
                o: o.to_host_vec(s)?,
                state: so.to_host_vec(s)?,
            })
        }

        fn norm(
            &self,
            fault: FaultSink,
            o: &[f32],
            z: &[f32],
            gain: &[f32],
            m: usize,
        ) -> Result<Vec<f32>, GateError> {
            let s = self.stream();
            let od = DeviceBuffer::from_host(s, o)?;
            let zd = DeviceBuffer::from_host(s, z)?;
            let wd = DeviceBuffer::from_host(s, gain)?;
            let mut y = DeviceBuffer::from_host(s, &vec![0.0f32; o.len()])?;
            self.k.norm_gate.enqueue_norm_gate(
                s,
                NormGateArgs {
                    o: &od,
                    z: &zd,
                    w: &wd,
                    eps: EPS,
                    n_v: SHAPE.n_v,
                    m,
                    fault,
                    y: &mut y,
                },
            )?;
            s.synchronize()?;
            Ok(y.to_host_vec(s)?)
        }
    }

    /// `same=N/M` and the largest ulp distance between two outputs.
    fn cmp(a: &[f32], b: &[f32]) -> String {
        format!(
            "same={}/{} max_ulp={}",
            same_bits(a, b),
            a.len().max(b.len()),
            max_ulps(a, b)
        )
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let k = LinearKernels::load(gpu.context())?;
        let cx = Ctx { gpu: &gpu, k: &k };
        println!(
            "gate_linear: device {} — n_k={} n_v={} head={HEAD} channels={} map={:?}, synthetic",
            gpu.device_name()?,
            SHAPE.n_k,
            SHAPE.n_v,
            SHAPE.channels(),
            SHAPE.map
        );
        let mut failed = 0u32;
        let mut clauses = 0u32;
        let mut tally = |pass: bool| {
            clauses += 1;
            failed += u32::from(!pass);
        };
        tally(math());
        for pass in conv_clauses(&cx)? {
            tally(pass);
        }
        for pass in delta_clauses(&cx)? {
            tally(pass);
        }
        for pass in norm_clauses(&cx)? {
            tally(pass);
        }
        tally(graph(&cx)?);
        tally(depth(&cx)?);
        for pass in fault_clauses(&cx)? {
            tally(pass);
        }
        tally(no_local_depot(&[
            "gdn_conv_prep",
            "gdn_delta",
            "gdn_norm_gate",
        ])?);
        let pass = failed == 0;
        println!(
            "gate_linear: {clauses} clauses, {failed} failed — {}",
            verdict(pass)
        );
        if !pass {
            return Err(checks_failed());
        }
        Ok(())
    }

    /// Clause 1.
    fn math() -> bool {
        let mut r = Lcg(0x6d61_7468);
        let xs: Vec<f32> = (0..1 << 20).map(|_| r.uniform(-87.0, 88.0)).collect();
        let e: Vec<f32> = xs.iter().map(|&x| expf_ik(x)).collect();
        let e64: Vec<f32> = xs.iter().map(|&x| f64::from(x).exp() as f32).collect();
        let us: Vec<f32> = (0..1 << 20)
            .map(|_| r.uniform(0.0, 20.0).exp())
            .chain([1.0, 1.0 + f32::EPSILON, 2.0, std::f32::consts::E])
            .collect();
        let l: Vec<f32> = us.iter().map(|&u| logf_poly(u)).collect();
        let l64: Vec<f32> = us.iter().map(|&u| f64::from(u).ln() as f32).collect();
        let (ue, ul) = (max_ulps(&e, &e64), max_ulps(&l, &l64));
        let pass = ue <= MATH_ULPS && ul <= MATH_ULPS;
        println!(
            "math expf_ik max_ulp={ue} over {} points, logf_poly max_ulp={ul} over {} points \
             (band {MATH_ULPS}) {}",
            xs.len(),
            us.len(),
            verdict(pass)
        );
        pass
    }

    /// Clause 2.
    fn conv_clauses(cx: &Ctx<'_>) -> Result<Vec<bool>, GateError> {
        let unl = cx.gpu.unlabelled_sink();
        let l = Layer::new(0x636f_6e76);
        let tk = Tokens::new(PROMPT, 0x746f_6b31);
        let s0 = Lcg(0x7330).fill(SHAPE.conv_state_len(), -1.0, 1.0);
        let mut out = Vec::new();
        for m in [PROMPT, 2, 1] {
            let t = tk.slice(0, m);
            let d = cx.conv(unl, &l, &t, &s0, m)?;
            let h = conv_prep_host(&t.x, &t.b, &t.a, &l.w, &l.dt, &l.sa, &s0, SHAPE, EPS, m);
            let pass = bits_equal(&d.y, &h.y)
                && bits_equal(&d.beta, &h.beta)
                && bits_equal(&d.decay, &h.decay)
                && bits_equal(&d.state, &h.state);
            println!(
                "conv m={m} y {} beta {} decay {} state {} {}",
                cmp(&d.y, &h.y),
                cmp(&d.beta, &h.beta),
                cmp(&d.decay, &h.decay),
                cmp(&d.state, &h.state),
                verdict(pass)
            );
            out.push(pass);
        }
        let whole = cx.conv(unl, &l, &tk, &s0, PROMPT)?;
        let mut state = s0;
        let (mut y, mut beta, mut decay) = (Vec::new(), Vec::new(), Vec::new());
        for t in 0..PROMPT {
            let d = cx.conv(unl, &l, &tk.slice(t, 1), &state, 1)?;
            y.extend_from_slice(&d.y);
            beta.extend_from_slice(&d.beta);
            decay.extend_from_slice(&d.decay);
            state = d.state;
        }
        let pass = bits_equal(&y, &whole.y)
            && bits_equal(&beta, &whole.beta)
            && bits_equal(&decay, &whole.decay)
            && bits_equal(&state, &whole.state);
        println!(
            "conv chain {PROMPT} one-token launches vs one {PROMPT}-token launch: y {} beta {} \
             decay {} state {} {}",
            cmp(&y, &whole.y),
            cmp(&beta, &whole.beta),
            cmp(&decay, &whole.decay),
            cmp(&state, &whole.state),
            verdict(pass)
        );
        out.push(pass);
        Ok(out)
    }

    /// `m` tokens of delta inputs from the host conv rule with a zero
    /// incoming conv state.
    fn host_step(m: usize, seed: u64) -> Step {
        let l = Layer::new(seed);
        let tk = Tokens::new(m, seed ^ 0x5eed);
        let zero = vec![0.0f32; SHAPE.conv_state_len()];
        Step::of(&conv_prep_host(
            &tk.x, &tk.b, &tk.a, &l.w, &l.dt, &l.sa, &zero, SHAPE, EPS, m,
        ))
    }

    fn delta_line(what: &str, d: &DeltaOut, h: &DeltaOut) -> bool {
        let pass = bits_equal(&d.o, &h.o) && bits_equal(&d.state, &h.state);
        println!(
            "delta {what}: o {} state {} {}",
            cmp(&d.o, &h.o),
            cmp(&d.state, &h.state),
            verdict(pass)
        );
        pass
    }

    /// Clause 3.
    fn delta_clauses(cx: &Ctx<'_>) -> Result<Vec<bool>, GateError> {
        let unl = cx.gpu.unlabelled_sink();
        let zero = vec![0.0f32; SHAPE.state_len()];
        let st = host_step(PROMPT + 1, 0x6465_6c74);
        let prompt = st.slice(0, PROMPT);
        let mut out = Vec::new();

        let d = cx.delta(unl, &prompt, &zero, SHAPE, PROMPT)?;
        let h = delta_host(
            &prompt.qkv,
            &prompt.beta,
            &prompt.decay,
            &zero,
            SHAPE,
            PROMPT,
        );
        out.push(delta_line(
            &format!("prompt m={PROMPT} from a zero state"),
            &d,
            &h,
        ));

        let step = st.slice(PROMPT, 1);
        let ds = cx.delta(unl, &step, &h.state, SHAPE, 1)?;
        let hs = delta_host(&step.qkv, &step.beta, &step.decay, &h.state, SHAPE, 1);
        out.push(delta_line(
            &format!("decode step at depth {PROMPT}"),
            &ds,
            &hs,
        ));

        let one = st.slice(0, 1);
        let d1 = cx.delta(unl, &one, &zero, SHAPE, 1)?;
        let h1 = delta_host(&one.qkv, &one.beta, &one.decay, &zero, SHAPE, 1);
        out.push(delta_line("prompt m=1 from a zero state", &d1, &h1));

        let grouped = LinearShape {
            map: KHeadMap::Grouped,
            ..SHAPE
        };
        let eight = st.slice(0, 8);
        let dg = cx.delta(unl, &eight, &zero, grouped, 8)?;
        let hg = delta_host(&eight.qkv, &eight.beta, &eight.decay, &zero, grouped, 8);
        out.push(delta_line("grouped key-head map m=8", &dg, &hg));

        let mut state = zero;
        let mut o = Vec::new();
        for t in 0..PROMPT {
            let dt = cx.delta(unl, &st.slice(t, 1), &state, SHAPE, 1)?;
            o.extend_from_slice(&dt.o);
            state = dt.state;
        }
        out.push(delta_line(
            &format!("chain {PROMPT} one-token launches vs one {PROMPT}-token launch"),
            &DeltaOut { o, state },
            &d,
        ));
        Ok(out)
    }

    /// Clause 4.
    fn norm_clauses(cx: &Ctx<'_>) -> Result<Vec<bool>, GateError> {
        let unl = cx.gpu.unlabelled_sink();
        let l = Layer::new(0x6e6f_726d);
        let mut r = Lcg(0x6f7a);
        let n = PROMPT * SHAPE.n_v * HEAD;
        let o = r.fill(n, -0.5, 0.5);
        let z = r.fill(n, -4.0, 4.0);
        let mut out = Vec::new();
        for m in [PROMPT, 1] {
            let len = m * SHAPE.n_v * HEAD;
            let d = cx.norm(unl, &o[..len], &z[..len], &l.gain, m)?;
            let h = norm_gate_host::<GATE_SILU>(&o[..len], &z[..len], &l.gain, EPS);
            let pass = bits_equal(&d, &h);
            println!("norm_gate m={m} y {} {}", cmp(&d, &h), verdict(pass));
            out.push(pass);
        }
        Ok(out)
    }

    /// One decode token's device inputs for the graph clause.
    struct TokenIns {
        x: DeviceBuffer<f32>,
        b: DeviceBuffer<f32>,
        a: DeviceBuffer<f32>,
        w: DeviceBuffer<f32>,
        dt: DeviceBuffer<f32>,
        sa: DeviceBuffer<f32>,
        conv_state: DeviceBuffer<f32>,
        state: DeviceBuffer<f32>,
        z: DeviceBuffer<f32>,
        gain: DeviceBuffer<f32>,
    }

    /// What the token's three launches write.
    struct TokenOuts {
        y: DeviceBuffer<f32>,
        beta: DeviceBuffer<f32>,
        decay: DeviceBuffer<f32>,
        conv_state: DeviceBuffer<f32>,
        o: DeviceBuffer<f32>,
        state: DeviceBuffer<f32>,
        out: DeviceBuffer<f32>,
    }

    impl TokenOuts {
        fn new(s: &CudaStream) -> Result<TokenOuts, GateError> {
            let (ch, nv) = (SHAPE.channels(), SHAPE.n_v);
            let z = |n: usize| DeviceBuffer::from_host(s, &vec![0.0f32; n]);
            Ok(TokenOuts {
                y: z(ch)?,
                beta: z(nv)?,
                decay: z(nv)?,
                conv_state: z(CONV_ROWS * ch)?,
                o: z(nv * HEAD)?,
                state: z(SHAPE.state_len())?,
                out: z(nv * HEAD)?,
            })
        }

        fn read(&self, s: &CudaStream) -> Result<Vec<Vec<f32>>, GateError> {
            Ok(vec![
                self.out.to_host_vec(s)?,
                self.state.to_host_vec(s)?,
                self.conv_state.to_host_vec(s)?,
            ])
        }
    }

    /// Enqueue one decode token's conv, delta and norm.
    fn enqueue_token(
        k: &LinearKernels,
        st: &CudaStream,
        fault: FaultSink,
        i: &TokenIns,
        o: &mut TokenOuts,
    ) -> Result<(), bloomery_gpu::GpuError> {
        k.conv.enqueue_conv_prep(
            st,
            ConvArgs {
                x: &i.x,
                b_raw: &i.b,
                a_raw: &i.a,
                w: &i.w,
                dt_bias: &i.dt,
                ssm_a: &i.sa,
                state_in: &i.conv_state,
                shape: SHAPE,
                eps: EPS,
                m: 1,
                fault,
                y: &mut o.y,
                beta: &mut o.beta,
                decay: &mut o.decay,
                state_out: &mut o.conv_state,
            },
        )?;
        k.delta.enqueue_delta(
            st,
            DeltaArgs {
                qkv: &o.y,
                beta: &o.beta,
                decay: &o.decay,
                state_in: &i.state,
                shape: SHAPE,
                m: 1,
                fault,
                o: &mut o.o,
                state_out: &mut o.state,
            },
        )?;
        k.norm_gate.enqueue_norm_gate(
            st,
            NormGateArgs {
                o: &o.o,
                z: &i.z,
                w: &i.gain,
                eps: EPS,
                n_v: SHAPE.n_v,
                m: 1,
                fault,
                y: &mut o.out,
            },
        )
    }

    /// Clause 5: conv, delta and norm of one token as a graph.
    fn graph(cx: &Ctx<'_>) -> Result<bool, GateError> {
        let s = cx.stream();
        let unl = cx.gpu.unlabelled_sink();
        let l = Layer::new(0x6772_6170);
        let tk = Tokens::new(1, 0x6731);
        let mut r = Lcg(0x6732);
        let up = |v: &[f32]| DeviceBuffer::from_host(s, v);
        let ins = TokenIns {
            x: up(&tk.x)?,
            b: up(&tk.b)?,
            a: up(&tk.a)?,
            w: up(&l.w)?,
            dt: up(&l.dt)?,
            sa: up(&l.sa)?,
            conv_state: up(&r.fill(SHAPE.conv_state_len(), -1.0, 1.0))?,
            state: up(&r.fill(SHAPE.state_len(), -0.1, 0.1))?,
            z: up(&r.fill(SHAPE.n_v * HEAD, -4.0, 4.0))?,
            gain: up(&l.gain)?,
        };
        let mut eager = TokenOuts::new(s)?;
        enqueue_token(cx.k, s, unl, &ins, &mut eager)?;
        s.synchronize()?;
        let mut replay = TokenOuts::new(s)?;
        let g = cx
            .gpu
            .capture(|st| enqueue_token(cx.k, st, unl, &ins, &mut replay))?;
        g.launch(s)?;
        s.synchronize()?;
        let nodes = g.node_count();
        let (a, b) = (eager.read(s)?, replay.read(s)?);
        let same = a.iter().zip(&b).all(|(x, y)| bits_equal(x, y));
        let pass = same && nodes == 3;
        println!(
            "graph conv+delta+norm_gate m=1: eager_vs_graph_bit_identical={same} graph_nodes={nodes} {}",
            verdict(pass)
        );
        drop(g);
        Ok(pass)
    }

    /// Clause 6.
    fn depth(cx: &Ctx<'_>) -> Result<bool, GateError> {
        let unl = cx.gpu.unlabelled_sink();
        let total = DEPTHS[DEPTHS.len() - 1];
        let l = Layer::new(0x6465_7074);
        let tk = Tokens::new(total, 0x6468);
        let mut cstate = vec![0.0f32; SHAPE.conv_state_len()];
        let mut state = vec![0.0f32; SHAPE.state_len()];
        let mut steps = Step {
            qkv: Vec::new(),
            beta: Vec::new(),
            decay: Vec::new(),
        };
        let mut at_depth = Vec::new();
        let mut t0 = 0;
        for &d in &DEPTHS {
            let m = d - t0;
            let c = cx.conv(unl, &l, &tk.slice(t0, m), &cstate, m)?;
            cstate = c.state.clone();
            let st = Step::of(&c);
            let dd = cx.delta(unl, &st, &state, SHAPE, m)?;
            state = dd.state;
            at_depth.push(state.clone());
            steps.qkv.extend_from_slice(&st.qkv);
            steps.beta.extend_from_slice(&st.beta);
            steps.decay.extend_from_slice(&st.decay);
            t0 = d;
        }
        let zero = vec![0.0f32; SHAPE.state_len()];
        let h = delta_host(&steps.qkv, &steps.beta, &steps.decay, &zero, SHAPE, total);
        let pass = bits_equal(&state, &h.state);
        println!(
            "depth {total} tokens in launches of {:?}: state vs the host rule over all {total} \
             at once {} {}",
            DEPTHS
                .iter()
                .scan(0, |p, &d| {
                    let m = d - *p;
                    *p = d;
                    Some(m)
                })
                .collect::<Vec<_>>(),
            cmp(&state, &h.state),
            verdict(pass)
        );
        let f64_states = f64_run(&steps, &DEPTHS);
        for ((&d, s32), s64) in DEPTHS.iter().zip(&at_depth).zip(&f64_states) {
            let big = s64.iter().fold(0.0f64, |a, v| a.max(v.abs()));
            let (mut worst, mut sq, mut ref_sq) = (0.0f64, 0.0f64, 0.0f64);
            for (&a, &b) in s32.iter().zip(s64) {
                let e = f64::from(a) - b;
                worst = worst.max(e.abs());
                sq += e * e;
                ref_sq += b * b;
            }
            println!(
                "drift depth={d} f32 state vs f64 run: max|d|/max|S|={:.3e} rms(d)/rms(S)={:.3e} \
                 max|S|={big:.3e} (reported, not judged)",
                worst / big,
                (sq / ref_sq).sqrt()
            );
        }
        Ok(pass)
    }

    /// The recurrence in f64 on the f32 inputs, one thread per head; the
    /// state at each depth.
    fn f64_run(st: &Step, depths: &[usize]) -> Vec<Vec<f64>> {
        let (n_k, n_v, ch) = (SHAPE.n_k, SHAPE.n_v, SHAPE.channels());
        let per_head: Vec<Vec<Vec<f64>>> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..n_v)
                .map(|h| {
                    sc.spawn(move || {
                        let kh = SHAPE.map.k_head(h, n_k, n_v);
                        let mut s = vec![0.0f64; HEAD * HEAD];
                        let mut snaps = Vec::new();
                        let mut t = 0;
                        for &d in depths {
                            while t < d {
                                let row = &st.qkv[t * ch..(t + 1) * ch];
                                let k = &row[(n_k + kh) * HEAD..(n_k + kh + 1) * HEAD];
                                let dc = f64::from(st.decay[t * n_v + h]);
                                let bt = f64::from(st.beta[t * n_v + h]);
                                for (col, sc) in s.chunks_mut(HEAD).enumerate() {
                                    let v = f64::from(row[2 * n_k * HEAD + h * HEAD + col]);
                                    let mut kv = 0.0f64;
                                    for (si, &ki) in sc.iter_mut().zip(k) {
                                        *si *= dc;
                                        kv += *si * f64::from(ki);
                                    }
                                    let u = (v - kv) * bt;
                                    for (si, &ki) in sc.iter_mut().zip(k) {
                                        *si += f64::from(ki) * u;
                                    }
                                }
                                t += 1;
                            }
                            snaps.push(s.clone());
                        }
                        snaps
                    })
                })
                .collect();
            hs.into_iter()
                .map(|h| h.join().expect("an f64 head thread panicked"))
                .collect()
        });
        (0..depths.len())
            .map(|i| per_head.iter().flat_map(|h| h[i].iter().copied()).collect())
            .collect()
    }

    /// One fault clause's verdict line: the word, which values changed from
    /// the clean launch, and that the word is clean around it.
    struct Planted<'a> {
        what: &'a str,
        want: Fault,
        before: Option<Fault>,
        word: Option<Fault>,
        after: Option<Fault>,
        /// Values that must be non-finite, in each output.
        hit: Vec<Vec<usize>>,
        clean: Vec<Vec<f32>>,
        bad: Vec<Vec<f32>>,
    }

    impl Planted<'_> {
        fn judge(&self) -> bool {
            let mut reach = true;
            let mut rest = true;
            let mut n_hit = 0;
            for ((hit, c), b) in self.hit.iter().zip(&self.clean).zip(&self.bad) {
                reach &= hit.iter().all(|&i| !b[i].is_finite());
                n_hit += hit.len();
                let mut mark = vec![false; b.len()];
                for &i in hit {
                    mark[i] = true;
                }
                rest &= c.len() == b.len()
                    && c.iter()
                        .zip(b)
                        .zip(&mark)
                        .all(|((x, y), &m)| m || x.to_bits() == y.to_bits());
            }
            let pass = self.before.is_none()
                && self.word == Some(self.want)
                && self.after.is_none()
                && reach
                && rest;
            println!(
                "fault {}: word \"{}\" (want \"{}\"), clean before {} and after {}, the {n_hit} \
                 values it reaches non-finite {reach}, every other value the clean launch's {rest} {}",
                self.what,
                self.word
                    .map_or_else(|| "none".to_owned(), |f| f.to_string()),
                self.want,
                self.before.is_none(),
                self.after.is_none(),
                verdict(pass)
            );
            pass
        }
    }

    /// Clause 7.
    fn fault_clauses(cx: &Ctx<'_>) -> Result<Vec<bool>, GateError> {
        let (ch, nv) = (SHAPE.channels(), SHAPE.n_v);
        let unl = cx.gpu.unlabelled_sink();
        let m = 16;
        let mut out = Vec::new();

        // Conv: a NaN in channel 100 (query head 0) of token 5, then 1e30 in
        // every channel of key head 3 at token 7 — finite, but its squares
        // overflow the norm.
        let l = Layer::new(0x6661_756c);
        let tk = Tokens::new(m, 0x6674);
        let cs = vec![0.0f32; SHAPE.conv_state_len()];
        let clean = cx.conv(unl, &l, &tk, &cs, m)?;
        let conv_outs = |c: &ConvOut| {
            vec![
                c.y.clone(),
                c.beta.clone(),
                c.decay.clone(),
                c.state.clone(),
            ]
        };
        let head_rows = |hp: usize, t0: usize| -> Vec<usize> {
            (t0..t0 + CONV_TAPS)
                .flat_map(|t| (0..HEAD).map(move |d| t * ch + hp * HEAD + d))
                .collect()
        };
        for (layer, what, plant, hp, t0) in [
            (21usize, "conv NaN input", 0usize, 0usize, 5usize),
            (
                22,
                "conv norm overflow from finite inputs",
                1,
                SHAPE.n_k + 3,
                7,
            ),
        ] {
            let sink = cx.gpu.layer_sink(layer)?;
            let before = cx.gpu.fault()?;
            let mut bad_tk = tk.slice(0, m);
            if plant == 0 {
                bad_tk.x[t0 * ch + 100] = f32::NAN;
            } else {
                for d in 0..HEAD {
                    bad_tk.x[t0 * ch + hp * HEAD + d] = 1e30;
                }
            }
            let bad = cx.conv(sink, &l, &bad_tk, &cs, m)?;
            let word = cx.gpu.take_fault()?;
            let _ = cx.conv(sink, &l, &tk, &cs, m)?;
            let after = cx.gpu.fault()?;
            out.push(
                Planted {
                    what,
                    want: Fault::at(u32::try_from(layer)?, FaultSite::LinearConv),
                    before,
                    word,
                    after,
                    hit: vec![head_rows(hp, t0), vec![], vec![], vec![]],
                    clean: conv_outs(&clean),
                    bad: conv_outs(&bad),
                }
                .judge(),
            );
        }

        // Delta: a NaN value (head 4, column 17, token 9); then a state
        // column (head 2, column 5) of 3e38 whose S'ᵀk overflows at token 0
        // (decay and β 1, the key all 1/√128).
        let mut st = host_step(m, 0x6664);
        let zero = vec![0.0f32; SHAPE.state_len()];
        let (h_nan, c_nan, t_nan) = (4usize, 17usize, 9usize);
        let (h_big, c_big) = (2usize, 5usize);
        let kh = SHAPE.map.k_head(h_big, SHAPE.n_k, nv);
        for d in 0..HEAD {
            st.qkv[(SHAPE.n_k + kh) * HEAD + d] = 1.0 / (HEAD as f32).sqrt();
        }
        st.decay[h_big] = 1.0;
        st.beta[h_big] = 1.0;
        let clean = cx.delta(unl, &st, &zero, SHAPE, m)?;
        let column = |h: usize, c: usize, t0: usize| -> (Vec<usize>, Vec<usize>) {
            (
                (t0..m).map(|t| (t * nv + h) * HEAD + c).collect(),
                (0..HEAD).map(|i| (h * HEAD + c) * HEAD + i).collect(),
            )
        };
        for (layer, what, big) in [
            (23usize, "delta NaN value", false),
            (24, "delta state overflow from finite inputs", true),
        ] {
            let sink = cx.gpu.layer_sink(layer)?;
            let before = cx.gpu.fault()?;
            let mut bst = st.slice(0, m);
            let mut si = zero.clone();
            let (ho, hs) = if big {
                for i in 0..HEAD {
                    si[(h_big * HEAD + c_big) * HEAD + i] = 3e38;
                }
                column(h_big, c_big, 0)
            } else {
                bst.qkv[t_nan * ch + 2 * SHAPE.n_k * HEAD + h_nan * HEAD + c_nan] = f32::NAN;
                column(h_nan, c_nan, t_nan)
            };
            let bad = cx.delta(sink, &bst, &si, SHAPE, m)?;
            let word = cx.gpu.take_fault()?;
            let _ = cx.delta(sink, &st, &zero, SHAPE, m)?;
            let after = cx.gpu.fault()?;
            out.push(
                Planted {
                    what,
                    want: Fault::at(u32::try_from(layer)?, FaultSite::LinearDelta),
                    before,
                    word,
                    after,
                    hit: vec![ho, hs],
                    clean: vec![clean.o.clone(), clean.state.clone()],
                    bad: vec![bad.o, bad.state],
                }
                .judge(),
            );
        }

        // Norm: a NaN gate value (token 3, head 7, value 40); then a head
        // (token 2, head 5) of 1e30, whose squares overflow the mean.
        let l = Layer::new(0x6667);
        let mut r = Lcg(0x6668);
        let n = m * nv * HEAD;
        let o = r.fill(n, -0.5, 0.5);
        let z = r.fill(n, -4.0, 4.0);
        let clean = cx.norm(unl, &o, &z, &l.gain, m)?;
        for (layer, what, big) in [
            (25usize, "norm_gate NaN gate", false),
            (26, "norm_gate mean overflow from finite inputs", true),
        ] {
            let sink = cx.gpu.layer_sink(layer)?;
            let before = cx.gpu.fault()?;
            let (mut bo, mut bz) = (o.clone(), z.clone());
            let hit: Vec<usize> = if big {
                let at = (2 * nv + 5) * HEAD;
                bo[at..at + HEAD].fill(1e30);
                (at..at + HEAD).collect()
            } else {
                let at = (3 * nv + 7) * HEAD + 40;
                bz[at] = f32::NAN;
                vec![at]
            };
            let bad = cx.norm(sink, &bo, &bz, &l.gain, m)?;
            let word = cx.gpu.take_fault()?;
            let _ = cx.norm(sink, &o, &z, &l.gain, m)?;
            let after = cx.gpu.fault()?;
            out.push(
                Planted {
                    what,
                    want: Fault::at(u32::try_from(layer)?, FaultSite::LinearGate),
                    before,
                    word,
                    after,
                    hit: vec![hit],
                    clean: vec![clean.clone()],
                    bad: vec![bad],
                }
                .judge(),
            );
        }
        Ok(out)
    }
}
