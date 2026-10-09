//! GPU gate for the linear-attention kernels (`bloomery_gpu::linear`), with
//! no model in its default run: synthetic layers and tokens at
//! Qwen3.6-35B-A3B's shapes (16 query/key heads, 32 value heads, 128 wide,
//! 8,192 conv channels, the tiled key-head map), GLM-5.3-Flash's KDA shape and
//! Qwen3.8's, from fixed seeds; the `kda_ik` case reads ik's GLM-5.3-Flash set
//! and the model file. Each kernel's output is held to its host rule bit for
//! bit; every clause draws its inputs from the host, not
//! from another kernel, so a defect turns only its own kernel's lines red.
//!
//! Clauses:
//! 1. `math`: [`expf_ik`] and [`logf_poly`] within [`MATH_ULPS`] of the f64
//!    functions (exp on [−87, 88], log on [1, 1 + e²⁰], the range softplus
//!    passes it).
//! 2. `conv`: the conv and prep of 512, 2 and 1 tokens at position 1,000
//!    over a random conv ring — output, β, decay and the ring after it; 512
//!    one-token launches chained through the ring equal to the 512-token
//!    launch; 8 tokens at the sequence start reading zeros over a random
//!    ring; and a call resuming from the ring at position 20, then a
//!    rollback into that pass, equal to the same tokens fed from position 0.
//! 3. `delta`: on a one-lane state read and written in place, a 512-token
//!    prompt from a zero state (`o` and the state after it), one decode step
//!    from that state, a one-token prompt from a zero state, the grouped
//!    key-head map on 8 tokens, and 512 one-token launches chained through
//!    the state equal to the 512-token launch; and the launcher's refusal of
//!    two lanes and of none.
//! 4. `norm_gate`: 512 and 1 tokens.
//! 5. `graph`: one decode token's conv, delta and norm captured once as a
//!    graph of three nodes and replayed for three successive positions, only
//!    the step's words rewritten between replays, equal to eager launches.
//! 6. `depth`: 4,096 tokens through the conv and the delta step on the card
//!    in launches of 6, 1,018 and 3,072 tokens, the state after the last
//!    equal to the host rule over all 4,096 at once; then (reported, not
//!    judged) how far the f32 state at depths 6, 1,024 and 4,096 sits from
//!    an f64 run of the same recurrence on the same inputs.
//! 7. `fault`: a NaN input and an overflow from finite inputs for each
//!    kernel, a position out of line for the conv and a lane word past the
//!    lanes for the delta step, each launched with a layer's sink: the word
//!    names that layer and the kernel's site, the values the planted input
//!    reaches are not finite, every other value is the clean launch's bit
//!    for bit, and the word is clean before and after.
//! 8. `shape`: the eight entries compile with no local depot.
//! 9. `kda` (Kimi Delta Attention at GLM-5.3-Flash's shape: 64 query/key and
//!    64 value heads, ε 1e-5, bound −5): `kda_conv_prep` at 512 and 1 tokens
//!    over a random ring against its host rule; its `y`, β and ring against
//!    `gdn_conv_prep`'s on the same channels, taps and β projection;
//!    `kda_delta` on a 512-token prompt from a zero state and one decode step
//!    after it against its host rule, and with every key at its head's decay
//!    against `gdn_delta`; `gdn_norm_gate_sigmoid` at 512 and 1 tokens; 4,096
//!    tokens through the KDA conv and delta in launches of 6, 1,018 and 3,072
//!    against the host rule over all at once; the launchers' refusals of a
//!    per-head-length decay and of a bound not below 0.
//! 10. `k1` (Qwen3.8's Gated DeltaNet: 16/48 heads, tiled, sigmoid gate): the
//!     GDN conv and delta on 512 tokens and the sigmoid-gated norm at 512 and 1
//!     tokens, each against its host rule.
//! 11. `kda fault`: a NaN forget value in the KDA conv, a NaN key decay in the
//!     KDA delta, a NaN gate in the sigmoid norm, judged as clause 7.
//! 12. `lanes` (`gdn_delta_lanes` at K1's shape over a state of
//!     [`ROW_LANES`] random lanes, the lane word on lane 2): the final mode
//!     of 8 and 1 tokens equal to the one-lane rule on lane 2 — `o`, lane 2
//!     after it, every other lane untouched, lane 2's stamp moved to the
//!     position after the call and the others kept; the row mode of 4 and 3
//!     tokens equal to that many one-token calls chained — `o`, row `j`'s
//!     state in lane `(2 + j) mod 4`, a lane no row wrote untouched, row
//!     `j`'s stamp its position plus one; lane 2's stamp not the call's first
//!     position raising `delta_stamp` with every value the clean launch's
//!     (judged as clause 7); and the launcher's refusal of row mode over more
//!     tokens than lanes.
//! 13. `kda lanes` (`kda_delta_lanes` at GLM-5.3-Flash's KDA shape over a
//!     state of [`KDA_LANES`] random lanes, the lane word on lane 1): row 0
//!     in place over 512 and 1 tokens equal to the one-lane rule on lane 1 —
//!     `o`, lane 1 after it, the other lane untouched, lane 1's stamp moved to
//!     the position after the call and the other kept; a verify of one token
//!     a row, row 0 in place then row 1 at row base 1, each its own launch on
//!     what the one before left, equal to two one-token calls chained — each
//!     row's `o`, row 0's state in lane 1 and row 1's in lane 0, stamps the
//!     positions after them; row 1 launched on a lane 1 stamped one past its
//!     position raising `delta_stamp` with every value the clean launch's
//!     (judged as clause 7); and the launcher's refusals of a row base at the
//!     lanes, a later row in place and row mode past the lanes.
//!
//! One case runs only when named, `--case kda_ik`, and alone: per KDA layer of
//! the `ref_glm5next` set, the KDA conv and prep on ik's own inputs against
//! ik's convolved, normed q·k·v, β and decay, the KDA delta on ik's own
//! inputs against `attn_output` and `new_state`, and the sigmoid norm on ik's
//! `attn_output` against `final_output`, in bands (its doc names the three
//! known differences). Without the glm5next reference family, or with it and a set,
//! tap or tensor missing, it fails by name.

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
    use bloomery_gpu::linear::conv::{
        ConvArgs, ConvOut, KdaConvArgs, conv_prep_host, kda_conv_prep_host,
    };
    use bloomery_gpu::linear::delta::{
        DeltaArgs, DeltaLanesArgs, DeltaOut, KdaLanesArgs, delta_host, kda_delta_host,
    };
    use bloomery_gpu::linear::norm_gate::{NormGateArgs, norm_gate_host};
    use bloomery_gpu::linear::{
        CONV_TAPS, GATE_SIGMOID, GATE_SILU, HEAD, KHeadMap, LinearKernels, LinearShape, PASS_ROWS,
        Q_SCALE, RING_ROWS, expf_ik, logf_poly, sigmoid,
    };
    use bloomery_gpu::{Fault, FaultSink, FaultSite, Gpu};
    use bloomery_gpu_gates::{
        GateError, RefManifest, bits_equal, checks_failed, max_ulps, no_local_depot,
        ref_tensor_logical_in, same_bits, split_f32, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::Split;

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
    /// The first position of the conv clauses past the sequence start.
    const P0: usize = 1000;
    /// The graph clause's first position and replays.
    const GRAPH_P0: usize = 40;
    const GRAPH_STEPS: usize = 3;
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
        shape: LinearShape,
        eps: f32,
        w: Vec<f32>,
        dt: Vec<f32>,
        sa: Vec<f32>,
        gain: Vec<f32>,
    }

    impl Layer {
        /// [`Layer::with`] at [`SHAPE`].
        fn new(seed: u64) -> Layer {
            Layer::with(seed, SHAPE)
        }

        /// Conv taps in [−0.5, 0.5); `ssm_a = −e^u`, `u` uniform over the
        /// file's range (−72.33 … −0.0186); `dt_bias` in [−1, 1); the norm
        /// gain in [0.5, 1) (the file's 0.52 … 0.96).
        fn with(seed: u64, shape: LinearShape) -> Layer {
            let mut r = Lcg(seed);
            let ch = shape.channels();
            let (lo, hi) = (0.0186f32.ln(), 72.33f32.ln());
            Layer {
                shape,
                eps: EPS,
                w: r.fill(CONV_TAPS * ch, -0.5, 0.5),
                dt: r.fill(shape.n_v, -1.0, 1.0),
                sa: (0..shape.n_v).map(|_| -r.uniform(lo, hi).exp()).collect(),
                gain: r.fill(HEAD, 0.5, 1.0),
            }
        }
    }

    /// `m` tokens of the projections the kernels read.
    struct Tokens {
        shape: LinearShape,
        x: Vec<f32>,
        b: Vec<f32>,
        a: Vec<f32>,
    }

    impl Tokens {
        fn new(m: usize, seed: u64) -> Tokens {
            Tokens::with(m, seed, SHAPE)
        }

        fn with(m: usize, seed: u64, shape: LinearShape) -> Tokens {
            let mut r = Lcg(seed);
            Tokens {
                shape,
                x: r.fill(m * shape.channels(), -1.0, 1.0),
                b: r.fill(m * shape.n_v, -3.0, 3.0),
                a: r.fill(m * shape.n_v, -3.0, 3.0),
            }
        }

        /// Tokens `t0 .. t0 + m`.
        fn slice(&self, t0: usize, m: usize) -> Tokens {
            let (ch, nv) = (self.shape.channels(), self.shape.n_v);
            Tokens {
                shape: self.shape,
                x: self.x[t0 * ch..(t0 + m) * ch].to_vec(),
                b: self.b[t0 * nv..(t0 + m) * nv].to_vec(),
                a: self.a[t0 * nv..(t0 + m) * nv].to_vec(),
            }
        }
    }

    /// The delta step's inputs for `m` tokens: the conv's outputs, `ch`
    /// channels, `nv` β values and `dn` decay values (`n_v`, or `n_v·HEAD`
    /// per key) a token.
    struct Step {
        qkv: Vec<f32>,
        beta: Vec<f32>,
        decay: Vec<f32>,
        ch: usize,
        nv: usize,
        dn: usize,
    }

    impl Step {
        /// The conv's outputs at `shape`, the decay's width read from their
        /// lengths.
        fn of(c: &ConvOut, shape: LinearShape) -> Step {
            let m = c.beta.len() / shape.n_v;
            Step {
                qkv: c.y.clone(),
                beta: c.beta.clone(),
                decay: c.decay.clone(),
                ch: shape.channels(),
                nv: shape.n_v,
                dn: c.decay.len() / m.max(1),
            }
        }

        /// No tokens yet, at `shape` with `dn` decay values a token.
        fn empty(shape: LinearShape, dn: usize) -> Step {
            Step {
                qkv: Vec::new(),
                beta: Vec::new(),
                decay: Vec::new(),
                ch: shape.channels(),
                nv: shape.n_v,
                dn,
            }
        }

        fn slice(&self, t0: usize, m: usize) -> Step {
            let (ch, nv, dn) = (self.ch, self.nv, self.dn);
            Step {
                qkv: self.qkv[t0 * ch..(t0 + m) * ch].to_vec(),
                beta: self.beta[t0 * nv..(t0 + m) * nv].to_vec(),
                decay: self.decay[t0 * dn..(t0 + m) * dn].to_vec(),
                ch,
                nv,
                dn,
            }
        }

        fn extend(&mut self, st: &Step) {
            self.qkv.extend_from_slice(&st.qkv);
            self.beta.extend_from_slice(&st.beta);
            self.decay.extend_from_slice(&st.decay);
        }
    }

    /// The decay's granularity a delta launch takes: `gdn_delta` or
    /// `kda_delta`.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Gran {
        Head,
        Key,
    }

    /// Positions `p0 .. p0 + m`.
    fn positions(p0: usize, m: usize) -> Vec<u32> {
        (p0..p0 + m).map(|p| p as u32).collect()
    }

    /// The single lane of every delta clause.
    const LANES: usize = 1;

    /// Lanes of the lane clauses' states: any count past one, four here.
    const ROW_LANES: usize = 4;

    /// Lanes of the KDA lane clause's state: GLM-5.3-Flash's, a verify of
    /// two rows.
    const KDA_LANES: usize = 2;

    /// A lanes launch's word and mode: the lane word, the call's first
    /// position, row mode.
    #[derive(Clone, Copy)]
    struct LaneCall {
        lane: u32,
        p0: u32,
        each: bool,
    }

    struct Ctx<'a> {
        gpu: &'a Gpu,
        k: &'a LinearKernels,
    }

    impl Ctx<'_> {
        fn stream(&self) -> &CudaStream {
            self.gpu.stream()
        }

        /// One conv launch of `m` tokens at `pos` over a ring holding
        /// `ring`; the outputs and the ring after it.
        fn conv(
            &self,
            fault: FaultSink,
            l: &Layer,
            tk: &Tokens,
            ring: &[f32],
            pos: &[u32],
            m: usize,
        ) -> Result<ConvOut, GateError> {
            let s = self.stream();
            let (ch, nv) = (l.shape.channels(), l.shape.n_v);
            let x = DeviceBuffer::from_host(s, &tk.x)?;
            let b = DeviceBuffer::from_host(s, &tk.b)?;
            let a = DeviceBuffer::from_host(s, &tk.a)?;
            let w = DeviceBuffer::from_host(s, &l.w)?;
            let dt = DeviceBuffer::from_host(s, &l.dt)?;
            let sa = DeviceBuffer::from_host(s, &l.sa)?;
            let pd = DeviceBuffer::from_host(s, pos)?;
            let mut y = DeviceBuffer::from_host(s, &vec![0.0f32; m * ch])?;
            let mut beta = DeviceBuffer::from_host(s, &vec![0.0f32; m * nv])?;
            let mut decay = DeviceBuffer::from_host(s, &vec![0.0f32; m * nv])?;
            let mut rd = DeviceBuffer::from_host(s, ring)?;
            self.k.conv.enqueue_conv_prep(
                s,
                ConvArgs {
                    x: &x,
                    b_raw: &b,
                    a_raw: &a,
                    w: &w,
                    dt_bias: &dt,
                    ssm_a: &sa,
                    pos: &pd,
                    shape: l.shape,
                    eps: l.eps,
                    m,
                    fault,
                    y: &mut y,
                    beta: &mut beta,
                    decay: &mut decay,
                    ring: &mut rd,
                },
            )?;
            s.synchronize()?;
            Ok(ConvOut {
                y: y.to_host_vec(s)?,
                beta: beta.to_host_vec(s)?,
                decay: decay.to_host_vec(s)?,
                ring: rd.to_host_vec(s)?,
            })
        }

        /// One delta launch of `m` tokens on a one-lane state holding
        /// `state`, with lane word `lane`; `o` and the state after it.
        #[allow(
            clippy::too_many_arguments,
            reason = "one launch's inputs, each as the launch names it"
        )]
        fn delta(
            &self,
            fault: FaultSink,
            st: &Step,
            state: &[f32],
            lane: u32,
            shape: LinearShape,
            gran: Gran,
            m: usize,
        ) -> Result<DeltaOut, GateError> {
            let s = self.stream();
            let qkv = DeviceBuffer::from_host(s, &st.qkv)?;
            let beta = DeviceBuffer::from_host(s, &st.beta)?;
            let decay = DeviceBuffer::from_host(s, &st.decay)?;
            let lw = DeviceBuffer::from_host(s, &[0xdead_beef, lane])?;
            let mut o = DeviceBuffer::from_host(s, &vec![0.0f32; m * shape.n_v * HEAD])?;
            let mut sd = DeviceBuffer::from_host(s, state)?;
            let args = DeltaArgs {
                qkv: &qkv,
                beta: &beta,
                decay: &decay,
                lane: &lw,
                lane_at: 1,
                lanes: LANES,
                shape,
                m,
                fault,
                o: &mut o,
                state: &mut sd,
            };
            match gran {
                Gran::Head => self.k.delta.enqueue_delta(s, args)?,
                Gran::Key => self.k.delta.enqueue_kda_delta(s, args)?,
            }
            s.synchronize()?;
            Ok(DeltaOut {
                o: o.to_host_vec(s)?,
                state: sd.to_host_vec(s)?,
            })
        }

        /// One `gdn_delta_lanes` launch of `m` tokens at K1's shape on a
        /// [`ROW_LANES`]-lane state holding `state` with `stamp`; `o`, the
        /// state and the stamps after it.
        fn delta_lanes(
            &self,
            fault: FaultSink,
            st: &Step,
            (state, stamp): (&[f32], &[u32]),
            call: LaneCall,
            m: usize,
        ) -> Result<(DeltaOut, Vec<u32>), GateError> {
            let s = self.stream();
            let qkv = DeviceBuffer::from_host(s, &st.qkv)?;
            let beta = DeviceBuffer::from_host(s, &st.beta)?;
            let decay = DeviceBuffer::from_host(s, &st.decay)?;
            let lw = DeviceBuffer::from_host(s, &[0xdead_beef, call.lane])?;
            let pd = DeviceBuffer::from_host(s, &[call.p0])?;
            let mut o = DeviceBuffer::from_host(s, &vec![0.0f32; m * K1_SHAPE.n_v * HEAD])?;
            let mut sd = DeviceBuffer::from_host(s, state)?;
            let mut sp = DeviceBuffer::from_host(s, stamp)?;
            self.k.delta.enqueue_delta_lanes(
                s,
                DeltaLanesArgs {
                    delta: DeltaArgs {
                        qkv: &qkv,
                        beta: &beta,
                        decay: &decay,
                        lane: &lw,
                        lane_at: 1,
                        lanes: ROW_LANES,
                        shape: K1_SHAPE,
                        m,
                        fault,
                        o: &mut o,
                        state: &mut sd,
                    },
                    each: call.each,
                    pos: &pd,
                    stamp: &mut sp,
                },
            )?;
            s.synchronize()?;
            Ok((
                DeltaOut {
                    o: o.to_host_vec(s)?,
                    state: sd.to_host_vec(s)?,
                },
                sp.to_host_vec(s)?,
            ))
        }

        /// One `kda_delta_lanes` launch of `m` tokens at the KDA shape on a
        /// [`KDA_LANES`]-lane state holding `state` with `stamp`, from verify
        /// row `row`; `o`, the state and the stamps after it.
        fn kda_lanes(
            &self,
            fault: FaultSink,
            st: &Step,
            (state, stamp): (&[f32], &[u32]),
            (call, row): (LaneCall, usize),
            m: usize,
        ) -> Result<(DeltaOut, Vec<u32>), GateError> {
            let s = self.stream();
            let qkv = DeviceBuffer::from_host(s, &st.qkv)?;
            let beta = DeviceBuffer::from_host(s, &st.beta)?;
            let decay = DeviceBuffer::from_host(s, &st.decay)?;
            let lw = DeviceBuffer::from_host(s, &[0xdead_beef, call.lane])?;
            let pd = DeviceBuffer::from_host(s, &[call.p0])?;
            let mut o = DeviceBuffer::from_host(s, &vec![0.0f32; m * KDA_SHAPE.n_v * HEAD])?;
            let mut sd = DeviceBuffer::from_host(s, state)?;
            let mut sp = DeviceBuffer::from_host(s, stamp)?;
            self.k.delta.enqueue_kda_delta_lanes(
                s,
                KdaLanesArgs {
                    lanes: DeltaLanesArgs {
                        delta: DeltaArgs {
                            qkv: &qkv,
                            beta: &beta,
                            decay: &decay,
                            lane: &lw,
                            lane_at: 1,
                            lanes: KDA_LANES,
                            shape: KDA_SHAPE,
                            m,
                            fault,
                            o: &mut o,
                            state: &mut sd,
                        },
                        each: call.each,
                        pos: &pd,
                        stamp: &mut sp,
                    },
                    row,
                },
            )?;
            s.synchronize()?;
            Ok((
                DeltaOut {
                    o: o.to_host_vec(s)?,
                    state: sd.to_host_vec(s)?,
                },
                sp.to_host_vec(s)?,
            ))
        }

        fn norm(
            &self,
            fault: FaultSink,
            o: &[f32],
            z: &[f32],
            gain: &[f32],
            m: usize,
        ) -> Result<Vec<f32>, GateError> {
            self.norm_act(fault, (o, z, gain), EPS, SHAPE.n_v, GATE_SILU, m)
        }

        /// The gated norm of `m·n_v` heads with the gate `act`
        /// ([`GATE_SILU`] or [`GATE_SIGMOID`]).
        fn norm_act(
            &self,
            fault: FaultSink,
            (o, z, gain): (&[f32], &[f32], &[f32]),
            eps: f32,
            n_v: usize,
            act: u32,
            m: usize,
        ) -> Result<Vec<f32>, GateError> {
            let s = self.stream();
            let od = DeviceBuffer::from_host(s, o)?;
            let zd = DeviceBuffer::from_host(s, z)?;
            let wd = DeviceBuffer::from_host(s, gain)?;
            let mut y = DeviceBuffer::from_host(s, &vec![0.0f32; o.len()])?;
            let args = NormGateArgs {
                o: &od,
                z: &zd,
                w: &wd,
                eps,
                n_v,
                m,
                fault,
                y: &mut y,
            };
            if act == GATE_SILU {
                self.k.norm_gate.enqueue_norm_gate(s, args)?;
            } else {
                self.k.norm_gate.enqueue_norm_gate_sigmoid(s, args)?;
            }
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
        let args: Vec<String> = std::env::args().skip(1).collect();
        let ik_only = match args.iter().map(String::as_str).collect::<Vec<_>>()[..] {
            [] => false,
            ["--case", "kda_ik"] => true,
            _ => return Err("usage: gate_linear [--case kda_ik]".into()),
        };
        let gpu = Gpu::new()?;
        let k = LinearKernels::load(gpu.context())?;
        let cx = Ctx { gpu: &gpu, k: &k };
        if ik_only {
            println!("gate_linear --case kda_ik: device {}", gpu.device_name()?);
            let pass = kda_ik_band(&cx)?;
            println!("gate_linear --case kda_ik: {}", verdict(pass));
            return if pass { Ok(()) } else { Err(checks_failed()) };
        }
        println!(
            "gate_linear: device {} — n_k={} n_v={} head={HEAD} channels={} map={:?} \
             ring={RING_ROWS} rows, synthetic",
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
            "kda_conv_prep",
            "kda_delta",
            "gdn_norm_gate_sigmoid",
            "gdn_delta_lanes",
            "kda_delta_lanes",
        ])?);
        for pass in kda_clauses(&cx)? {
            tally(pass);
        }
        for pass in k1_clauses(&cx)? {
            tally(pass);
        }
        for pass in kda_fault_clauses(&cx)? {
            tally(pass);
        }
        for pass in lanes_clauses(&cx)? {
            tally(pass);
        }
        for pass in kda_lanes_clauses(&cx)? {
            tally(pass);
        }
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

    /// The host rule's conv outputs for `tk` at `pos` over `ring`.
    fn conv_host(l: &Layer, tk: &Tokens, ring: &[f32], pos: &[u32], m: usize) -> ConvOut {
        conv_prep_host(
            &tk.x, &tk.b, &tk.a, &l.w, &l.dt, &l.sa, pos, ring, l.shape, l.eps, m,
        )
    }

    /// Whether every output of two conv runs is bit-identical, and the line
    /// fragment that says so.
    fn conv_same(d: &ConvOut, h: &ConvOut) -> (bool, String) {
        let pass = bits_equal(&d.y, &h.y)
            && bits_equal(&d.beta, &h.beta)
            && bits_equal(&d.decay, &h.decay)
            && bits_equal(&d.ring, &h.ring);
        (
            pass,
            format!(
                "y {} beta {} decay {} ring {}",
                cmp(&d.y, &h.y),
                cmp(&d.beta, &h.beta),
                cmp(&d.decay, &h.decay),
                cmp(&d.ring, &h.ring)
            ),
        )
    }

    /// Clause 2.
    fn conv_clauses(cx: &Ctx<'_>) -> Result<Vec<bool>, GateError> {
        let unl = cx.gpu.unlabelled_sink();
        let ch = SHAPE.channels();
        let l = Layer::new(0x636f_6e76);
        let tk = Tokens::new(PROMPT, 0x746f_6b31);
        let dirty = Lcg(0x7330).fill(SHAPE.ring_len(), -1.0, 1.0);
        let zero_ring = vec![0.0f32; SHAPE.ring_len()];
        let mut out = Vec::new();

        // Every token past the sequence start: the first three read the
        // ring, and a 512-token call's last rows overwrite the slots they
        // read.
        for m in [PROMPT, 2, 1] {
            let t = tk.slice(0, m);
            let pos = positions(P0, m);
            let d = cx.conv(unl, &l, &t, &dirty, &pos, m)?;
            let h = conv_host(&l, &t, &dirty, &pos, m);
            let (pass, line) = conv_same(&d, &h);
            println!(
                "conv m={m} at p0={P0} over a random ring: {line} {}",
                verdict(pass)
            );
            out.push(pass);
        }

        let pos = positions(P0, PROMPT);
        let whole = cx.conv(unl, &l, &tk, &dirty, &pos, PROMPT)?;
        let mut ring = dirty.clone();
        let (mut y, mut beta, mut decay) = (Vec::new(), Vec::new(), Vec::new());
        for t in 0..PROMPT {
            let d = cx.conv(unl, &l, &tk.slice(t, 1), &ring, &pos[t..=t], 1)?;
            y.extend_from_slice(&d.y);
            beta.extend_from_slice(&d.beta);
            decay.extend_from_slice(&d.decay);
            ring = d.ring;
        }
        let (pass, line) = conv_same(
            &ConvOut {
                y,
                beta,
                decay,
                ring,
            },
            &whole,
        );
        println!(
            "conv chain {PROMPT} one-token launches vs one {PROMPT}-token launch at p0={P0}: {line} {}",
            verdict(pass)
        );
        out.push(pass);

        // The sequence start reads zeros whatever the ring holds.
        let m = PASS_ROWS;
        let t = tk.slice(0, m);
        let pos = positions(0, m);
        let d = cx.conv(unl, &l, &t, &dirty, &pos, m)?;
        let h = conv_host(&l, &t, &dirty, &pos, m);
        let z = cx.conv(unl, &l, &t, &zero_ring, &pos, m)?;
        let (same_h, line) = conv_same(&d, &h);
        let pass = same_h && bits_equal(&d.y, &z.y);
        println!(
            "conv start m={m} at p0=0 over a random ring: {line}; y vs over a zero ring {} {}",
            cmp(&d.y, &z.y),
            verdict(pass)
        );
        out.push(pass);

        // A call that resumes from the ring, then a rollback into a full
        // pass: each equal to the same tokens fed from position 0 in one
        // call. The pass writes PASS_ROWS slots; the rollback to its second
        // position reads the slot of the position three before it, which
        // the call before the pass wrote.
        let (a, b, back) = (20usize, PASS_ROWS, 21usize);
        let fresh = Tokens::new(4, 0x7472_6b32);
        let ta = tk.slice(0, a);
        let tb = tk.slice(a, b);
        let da = cx.conv(unl, &l, &ta, &dirty, &positions(0, a), a)?;
        let db = cx.conv(unl, &l, &tb, &da.ring, &positions(a, b), b)?;
        let dc = cx.conv(unl, &l, &fresh, &db.ring, &positions(back, 4), 4)?;
        let ref_ab = conv_host(
            &l,
            &tk.slice(0, a + b),
            &zero_ring,
            &positions(0, a + b),
            a + b,
        );
        let joined = Tokens {
            shape: SHAPE,
            x: [&tk.x[..back * ch], &fresh.x[..]].concat(),
            b: [&tk.b[..back * SHAPE.n_v], &fresh.b[..]].concat(),
            a: [&tk.a[..back * SHAPE.n_v], &fresh.a[..]].concat(),
        };
        let ref_c = conv_host(&l, &joined, &zero_ring, &positions(0, back + 4), back + 4);
        let (want_b, want_c) = (&ref_ab.y[a * ch..], &ref_c.y[back * ch..]);
        let pass = bits_equal(&db.y, want_b) && bits_equal(&dc.y, want_c);
        println!(
            "conv resume at p0={a} (m={b}) after a {a}-token call: y vs the same tokens fed from 0 \
             {}; rollback to p0={back} (m=4, new tokens) after it: y {} {}",
            cmp(&db.y, want_b),
            cmp(&dc.y, want_c),
            verdict(pass)
        );
        out.push(pass);
        Ok(out)
    }

    /// `m` tokens of delta inputs from the host conv rule at the sequence
    /// start.
    fn host_step(m: usize, seed: u64) -> Step {
        let l = Layer::new(seed);
        let tk = Tokens::new(m, seed ^ 0x5eed);
        let zero = vec![0.0f32; SHAPE.ring_len()];
        Step::of(&conv_host(&l, &tk, &zero, &positions(0, m), m), SHAPE)
    }

    /// The host rule of one delta call on the one-lane state `state`.
    fn delta_ref(st: &Step, state: &[f32], shape: LinearShape, m: usize) -> DeltaOut {
        delta_host(&st.qkv, &st.beta, &st.decay, state, LANES, 0, shape, m)
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

        let d = cx.delta(unl, &prompt, &zero, 0, SHAPE, Gran::Head, PROMPT)?;
        let h = delta_ref(&prompt, &zero, SHAPE, PROMPT);
        out.push(delta_line(
            &format!("prompt m={PROMPT} from a zero state"),
            &d,
            &h,
        ));

        let step = st.slice(PROMPT, 1);
        let ds = cx.delta(unl, &step, &h.state, 0, SHAPE, Gran::Head, 1)?;
        let hs = delta_ref(&step, &h.state, SHAPE, 1);
        out.push(delta_line(
            &format!("decode step at depth {PROMPT}"),
            &ds,
            &hs,
        ));

        let one = st.slice(0, 1);
        let d1 = cx.delta(unl, &one, &zero, 0, SHAPE, Gran::Head, 1)?;
        let h1 = delta_ref(&one, &zero, SHAPE, 1);
        out.push(delta_line("prompt m=1 from a zero state", &d1, &h1));

        let grouped = LinearShape {
            map: KHeadMap::Grouped,
            ..SHAPE
        };
        let eight = st.slice(0, 8);
        let dg = cx.delta(unl, &eight, &zero, 0, grouped, Gran::Head, 8)?;
        let hg = delta_ref(&eight, &zero, grouped, 8);
        out.push(delta_line("grouped key-head map m=8", &dg, &hg));

        let mut state = zero;
        let mut o = Vec::new();
        for t in 0..PROMPT {
            let dt = cx.delta(unl, &st.slice(t, 1), &state, 0, SHAPE, Gran::Head, 1)?;
            o.extend_from_slice(&dt.o);
            state = dt.state;
        }
        out.push(delta_line(
            &format!("chain {PROMPT} one-token launches vs one {PROMPT}-token launch"),
            &DeltaOut { o, state },
            &d,
        ));

        out.push(lanes_refused(cx, &one)?);
        Ok(out)
    }

    /// The launcher refuses a state of more than one lane, and of none, by
    /// name.
    fn lanes_refused(cx: &Ctx<'_>, st: &Step) -> Result<bool, GateError> {
        let s = cx.stream();
        let unl = cx.gpu.unlabelled_sink();
        let qkv = DeviceBuffer::from_host(s, &st.qkv)?;
        let beta = DeviceBuffer::from_host(s, &st.beta)?;
        let decay = DeviceBuffer::from_host(s, &st.decay)?;
        let lw = DeviceBuffer::from_host(s, &[0u32])?;
        let mut o = DeviceBuffer::from_host(s, &vec![0.0f32; SHAPE.n_v * HEAD])?;
        let mut state = DeviceBuffer::from_host(s, &vec![0.0f32; 2 * SHAPE.state_len()])?;
        let mut said = Vec::new();
        for lanes in [2usize, 0] {
            let r = cx.k.delta.enqueue_delta(
                s,
                DeltaArgs {
                    qkv: &qkv,
                    beta: &beta,
                    decay: &decay,
                    lane: &lw,
                    lane_at: 0,
                    lanes,
                    shape: SHAPE,
                    m: 1,
                    fault: unl,
                    o: &mut o,
                    state: &mut state,
                },
            );
            said.push(match r {
                Ok(()) => (false, format!("lanes={lanes}: launched")),
                Err(e) => {
                    let e = e.to_string();
                    (e.contains(&format!("lanes={lanes}")), e)
                }
            });
        }
        s.synchronize()?;
        let pass = said.iter().all(|(named, _)| *named);
        println!(
            "delta launcher refuses lanes=2: \"{}\" and lanes=0: \"{}\" {}",
            said[0].1,
            said[1].1,
            verdict(pass)
        );
        Ok(pass)
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

    /// One decode token's device inputs for the graph clause: the step's
    /// words (the token's projections and its position) are rewritten in
    /// place before every launch; the lane word and the layer stay.
    struct TokenIns {
        x: DeviceBuffer<f32>,
        b: DeviceBuffer<f32>,
        a: DeviceBuffer<f32>,
        pos: DeviceBuffer<u32>,
        lane: DeviceBuffer<u32>,
        w: DeviceBuffer<f32>,
        dt: DeviceBuffer<f32>,
        sa: DeviceBuffer<f32>,
        z: DeviceBuffer<f32>,
        gain: DeviceBuffer<f32>,
    }

    /// One chain's state (read and written in place) and what the token's
    /// three launches write.
    struct TokenOuts {
        ring: DeviceBuffer<f32>,
        state: DeviceBuffer<f32>,
        y: DeviceBuffer<f32>,
        beta: DeviceBuffer<f32>,
        decay: DeviceBuffer<f32>,
        o: DeviceBuffer<f32>,
        out: DeviceBuffer<f32>,
    }

    impl TokenOuts {
        fn new(s: &CudaStream, ring: &[f32], state: &[f32]) -> Result<TokenOuts, GateError> {
            let (ch, nv) = (SHAPE.channels(), SHAPE.n_v);
            let z = |n: usize| DeviceBuffer::from_host(s, &vec![0.0f32; n]);
            Ok(TokenOuts {
                ring: DeviceBuffer::from_host(s, ring)?,
                state: DeviceBuffer::from_host(s, state)?,
                y: z(ch)?,
                beta: z(nv)?,
                decay: z(nv)?,
                o: z(nv * HEAD)?,
                out: z(nv * HEAD)?,
            })
        }

        fn read(&self, s: &CudaStream) -> Result<Vec<Vec<f32>>, GateError> {
            Ok(vec![
                self.out.to_host_vec(s)?,
                self.state.to_host_vec(s)?,
                self.ring.to_host_vec(s)?,
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
                pos: &i.pos,
                shape: SHAPE,
                eps: EPS,
                m: 1,
                fault,
                y: &mut o.y,
                beta: &mut o.beta,
                decay: &mut o.decay,
                ring: &mut o.ring,
            },
        )?;
        k.delta.enqueue_delta(
            st,
            DeltaArgs {
                qkv: &o.y,
                beta: &o.beta,
                decay: &o.decay,
                lane: &i.lane,
                lane_at: 0,
                lanes: LANES,
                shape: SHAPE,
                m: 1,
                fault,
                o: &mut o.o,
                state: &mut o.state,
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

    /// Clause 5: conv, delta and norm of one token captured once as a graph,
    /// then replayed for [`GRAPH_STEPS`] successive tokens with only the
    /// step's words rewritten between replays.
    fn graph(cx: &Ctx<'_>) -> Result<bool, GateError> {
        let s = cx.stream();
        let unl = cx.gpu.unlabelled_sink();
        let l = Layer::new(0x6772_6170);
        let tk = Tokens::new(GRAPH_STEPS, 0x6731);
        let mut r = Lcg(0x6732);
        let ring = r.fill(SHAPE.ring_len(), -1.0, 1.0);
        let state = r.fill(SHAPE.state_len(), -0.1, 0.1);
        let up = |v: &[f32]| DeviceBuffer::from_host(s, v);
        let one = tk.slice(0, 1);
        let mut ins = TokenIns {
            x: up(&one.x)?,
            b: up(&one.b)?,
            a: up(&one.a)?,
            pos: DeviceBuffer::from_host(s, &[0u32])?,
            lane: DeviceBuffer::from_host(s, &[0u32])?,
            w: up(&l.w)?,
            dt: up(&l.dt)?,
            sa: up(&l.sa)?,
            z: up(&r.fill(SHAPE.n_v * HEAD, -4.0, 4.0))?,
            gain: up(&l.gain)?,
        };
        let step = |ins: &mut TokenIns, t: usize| -> Result<(), GateError> {
            let tt = tk.slice(t, 1);
            ins.x.copy_from_host(s, &tt.x)?;
            ins.b.copy_from_host(s, &tt.b)?;
            ins.a.copy_from_host(s, &tt.a)?;
            ins.pos.copy_from_host(s, &[u32::try_from(GRAPH_P0 + t)?])?;
            Ok(())
        };
        let mut eager = TokenOuts::new(s, &ring, &state)?;
        let mut eager_rows = Vec::new();
        for t in 0..GRAPH_STEPS {
            step(&mut ins, t)?;
            enqueue_token(cx.k, s, unl, &ins, &mut eager)?;
            s.synchronize()?;
            eager_rows.push(eager.read(s)?);
        }
        let mut replay = TokenOuts::new(s, &ring, &state)?;
        let g = cx
            .gpu
            .capture(|st| enqueue_token(cx.k, st, unl, &ins, &mut replay))?;
        let mut same = true;
        for (t, want) in eager_rows.iter().enumerate() {
            step(&mut ins, t)?;
            g.launch(s)?;
            s.synchronize()?;
            let got = replay.read(s)?;
            same &= got.iter().zip(want).all(|(x, y)| bits_equal(x, y));
        }
        let nodes = g.node_count();
        let pass = same && nodes == 3;
        println!(
            "graph conv+delta+norm_gate m=1 captured once, replayed at positions {GRAPH_P0}..{}: \
             eager_vs_graph_bit_identical={same} graph_nodes={nodes} {}",
            GRAPH_P0 + GRAPH_STEPS - 1,
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
        let mut ring = vec![0.0f32; SHAPE.ring_len()];
        let mut state = vec![0.0f32; SHAPE.state_len()];
        let mut steps = Step::empty(SHAPE, SHAPE.n_v);
        let mut at_depth = Vec::new();
        let mut t0 = 0;
        for &d in &DEPTHS {
            let m = d - t0;
            let c = cx.conv(unl, &l, &tk.slice(t0, m), &ring, &positions(t0, m), m)?;
            ring = c.ring.clone();
            let st = Step::of(&c, SHAPE);
            let dd = cx.delta(unl, &st, &state, 0, SHAPE, Gran::Head, m)?;
            state = dd.state;
            at_depth.push(state.clone());
            steps.extend(&st);
            t0 = d;
        }
        let zero = vec![0.0f32; SHAPE.state_len()];
        let h = delta_ref(&steps, &zero, SHAPE, total);
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

        // Conv, at p0 = 100 over a random ring: a NaN in channel 100 (query
        // head 0) of token 2, then 1e30 in every channel of key head 3 at
        // token 3 — finite, but its squares overflow the norm; both below
        // the rows the call writes into the ring. Then token 1's position
        // one past p0 + 1.
        let l = Layer::new(0x6661_756c);
        let tk = Tokens::new(m, 0x6674);
        let ring = Lcg(0x6672).fill(SHAPE.ring_len(), -1.0, 1.0);
        let pos = positions(100, m);
        let clean = cx.conv(unl, &l, &tk, &ring, &pos, m)?;
        let conv_outs =
            |c: &ConvOut| vec![c.y.clone(), c.beta.clone(), c.decay.clone(), c.ring.clone()];
        let head_rows = |hp: usize, t0: usize| -> Vec<usize> {
            (t0..t0 + CONV_TAPS)
                .flat_map(|t| (0..HEAD).map(move |d| t * ch + hp * HEAD + d))
                .collect()
        };
        for (layer, what, plant, hp, t0) in [
            (21usize, "conv NaN input", 0usize, 0usize, 2usize),
            (
                22,
                "conv norm overflow from finite inputs",
                1,
                SHAPE.n_k + 3,
                3,
            ),
            (27, "conv position not p0 + t", 2, 0, 1),
        ] {
            let sink = cx.gpu.layer_sink(layer)?;
            let before = cx.gpu.fault()?;
            let mut bad_tk = tk.slice(0, m);
            let mut bad_pos = pos.clone();
            let hit = match plant {
                0 => {
                    bad_tk.x[t0 * ch + 100] = f32::NAN;
                    head_rows(hp, t0)
                }
                1 => {
                    for d in 0..HEAD {
                        bad_tk.x[t0 * ch + hp * HEAD + d] = 1e30;
                    }
                    head_rows(hp, t0)
                }
                _ => {
                    bad_pos[t0] += 1;
                    (t0 * ch..(t0 + 1) * ch).collect()
                }
            };
            let bad = cx.conv(sink, &l, &bad_tk, &ring, &bad_pos, m)?;
            let word = cx.gpu.take_fault()?;
            let _ = cx.conv(sink, &l, &tk, &ring, &pos, m)?;
            let after = cx.gpu.fault()?;
            out.push(
                Planted {
                    what,
                    want: Fault::at(u32::try_from(layer)?, FaultSite::LinearConv),
                    before,
                    word,
                    after,
                    hit: vec![hit, vec![], vec![], vec![]],
                    clean: conv_outs(&clean),
                    bad: conv_outs(&bad),
                }
                .judge(),
            );
        }

        // Delta: a NaN value (head 4, column 17, token 9); then a state
        // column (head 2, column 5) of 3e38 whose S'ᵀk overflows at token 0
        // (decay and β 1, the key all 1/√128); then the lane word 1 of a
        // one-lane state, which reads lane 0 and changes nothing else.
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
        let clean = cx.delta(unl, &st, &zero, 0, SHAPE, Gran::Head, m)?;
        let column = |h: usize, c: usize, t0: usize| -> (Vec<usize>, Vec<usize>) {
            (
                (t0..m).map(|t| (t * nv + h) * HEAD + c).collect(),
                (0..HEAD).map(|i| (h * HEAD + c) * HEAD + i).collect(),
            )
        };
        for (layer, what, plant, site) in [
            (23usize, "delta NaN value", 0usize, FaultSite::LinearDelta),
            (
                24,
                "delta state overflow from finite inputs",
                1,
                FaultSite::LinearDelta,
            ),
            (
                28,
                "delta lane word 1 of a one-lane state",
                2,
                FaultSite::DeltaLane,
            ),
        ] {
            let sink = cx.gpu.layer_sink(layer)?;
            let before = cx.gpu.fault()?;
            let mut bst = st.slice(0, m);
            let mut si = zero.clone();
            let mut lane = 0;
            let (ho, hs) = match plant {
                0 => {
                    bst.qkv[t_nan * ch + 2 * SHAPE.n_k * HEAD + h_nan * HEAD + c_nan] = f32::NAN;
                    column(h_nan, c_nan, t_nan)
                }
                1 => {
                    for i in 0..HEAD {
                        si[(h_big * HEAD + c_big) * HEAD + i] = 3e38;
                    }
                    column(h_big, c_big, 0)
                }
                _ => {
                    lane = 1;
                    (vec![], vec![])
                }
            };
            let bad = cx.delta(sink, &bst, &si, lane, SHAPE, Gran::Head, m)?;
            let word = cx.gpu.take_fault()?;
            let _ = cx.delta(sink, &st, &zero, 0, SHAPE, Gran::Head, m)?;
            let after = cx.gpu.fault()?;
            out.push(
                Planted {
                    what,
                    want: Fault::at(u32::try_from(layer)?, site),
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

    /// GLM-5.3-Flash's KDA layers: 64 query/key and 64 value heads.
    const KDA_SHAPE: LinearShape = LinearShape {
        n_k: 64,
        n_v: 64,
        map: KHeadMap::Tiled,
    };
    /// GLM-5.3-Flash's `layer_norm_rms_epsilon`: the KDA layers' L2 norm and
    /// gated norm.
    const KDA_EPS: f32 = 1e-5;
    /// GLM-5.3-Flash's `kda.gate_lower_bound`.
    const KDA_LB: f32 = -5.0;
    /// Qwen3.8-Flash-Next's linear layers (K1): 16 query/key heads, 48 value
    /// heads, the tiled map.
    const K1_SHAPE: LinearShape = LinearShape {
        n_k: 16,
        n_v: 48,
        map: KHeadMap::Tiled,
    };
    /// The ik band's bounds, `‖ours − ik‖ / ‖ik‖` over a tap, teacher-forced
    /// on ik's own inputs of one layer over the batch set's 5 tokens from a
    /// zero state.
    // PIN(2026-09-27): per token the known differences (a 128-term dot in two
    // orders, ~√128·2⁻²⁴ ≈ 7e-7 relative; libm expf against expf_ik, ≤ 2 ulp
    // on β and each decay; q scaled before against the output after) are
    // ~1e-6; the recurrence contracts (decay ≤ 1, ‖k‖ = 1, β < 1), so over the
    // set's 5 tokens the error adds at most linearly, ≤ 5 × 1e-6 = 5e-6
    // [derived]; 1e-4 is 20× that, a loose band. The gated norm takes ik's
    // own attn_output: one f64 mean and one sigmoid apart, ~3e-7 [derived],
    // 1e-5.
    const IK_BAND_DELTA: f64 = 1e-4;
    const IK_BAND_NORM: f64 = 1e-5;

    /// One KDA layer's parameters: conv taps, the per-key `dt_bias`
    /// (`[n_v·HEAD]`), `ssm_a = −e^A_log` per head and the norm gain.
    struct KdaLayer {
        w: Vec<f32>,
        dt: Vec<f32>,
        sa: Vec<f32>,
        gain: Vec<f32>,
    }

    impl KdaLayer {
        /// Taps in [−0.5, 0.5); `dt_bias` in [−1, 1); `e^A_log` log-uniform
        /// on [0.02, 20]; the gain in [0.5, 1).
        fn new(seed: u64) -> KdaLayer {
            let mut r = Lcg(seed);
            let (ch, nv) = (KDA_SHAPE.channels(), KDA_SHAPE.n_v);
            let (lo, hi) = (0.02f32.ln(), 20.0f32.ln());
            KdaLayer {
                w: r.fill(CONV_TAPS * ch, -0.5, 0.5),
                dt: r.fill(nv * HEAD, -1.0, 1.0),
                sa: (0..nv).map(|_| -r.uniform(lo, hi).exp()).collect(),
                gain: r.fill(HEAD, 0.5, 1.0),
            }
        }
    }

    /// `m` tokens of a KDA layer's projections: the channels, β's raw
    /// projection and the low-rank forget projection `f` (`[m][n_v·HEAD]`).
    struct KdaTokens {
        x: Vec<f32>,
        b: Vec<f32>,
        f: Vec<f32>,
    }

    impl KdaTokens {
        fn new(m: usize, seed: u64) -> KdaTokens {
            let mut r = Lcg(seed);
            let (ch, nv) = (KDA_SHAPE.channels(), KDA_SHAPE.n_v);
            KdaTokens {
                x: r.fill(m * ch, -1.0, 1.0),
                b: r.fill(m * nv, -3.0, 3.0),
                f: r.fill(m * nv * HEAD, -3.0, 3.0),
            }
        }

        fn slice(&self, t0: usize, m: usize) -> KdaTokens {
            let (ch, nv) = (KDA_SHAPE.channels(), KDA_SHAPE.n_v);
            KdaTokens {
                x: self.x[t0 * ch..(t0 + m) * ch].to_vec(),
                b: self.b[t0 * nv..(t0 + m) * nv].to_vec(),
                f: self.f[t0 * nv * HEAD..(t0 + m) * nv * HEAD].to_vec(),
            }
        }
    }

    impl Ctx<'_> {
        /// One KDA conv launch of `m` tokens at `pos` over a ring holding
        /// `ring`, with the decay's bound `lb`; the outputs and the ring.
        #[allow(
            clippy::too_many_arguments,
            reason = "one launch's inputs, each as the launch names it"
        )]
        fn kda_conv(
            &self,
            fault: FaultSink,
            l: &KdaLayer,
            tk: &KdaTokens,
            ring: &[f32],
            pos: &[u32],
            lb: f32,
            m: usize,
        ) -> Result<ConvOut, GateError> {
            let s = self.stream();
            let (ch, nv) = (KDA_SHAPE.channels(), KDA_SHAPE.n_v);
            let up = |v: &[f32]| DeviceBuffer::from_host(s, v);
            let (x, b, f) = (up(&tk.x)?, up(&tk.b)?, up(&tk.f)?);
            let (w, dt, sa) = (up(&l.w)?, up(&l.dt)?, up(&l.sa)?);
            let pd = DeviceBuffer::from_host(s, pos)?;
            let mut y = up(&vec![0.0f32; m * ch])?;
            let mut beta = up(&vec![0.0f32; m * nv])?;
            let mut decay = up(&vec![0.0f32; m * nv * HEAD])?;
            let mut rd = up(ring)?;
            self.k.conv.enqueue_kda_conv_prep(
                s,
                KdaConvArgs {
                    x: &x,
                    b_raw: &b,
                    f: &f,
                    w: &w,
                    dt_bias: &dt,
                    ssm_a: &sa,
                    pos: &pd,
                    shape: KDA_SHAPE,
                    lb,
                    eps: KDA_EPS,
                    m,
                    fault,
                    y: &mut y,
                    beta: &mut beta,
                    decay: &mut decay,
                    ring: &mut rd,
                },
            )?;
            s.synchronize()?;
            Ok(ConvOut {
                y: y.to_host_vec(s)?,
                beta: beta.to_host_vec(s)?,
                decay: decay.to_host_vec(s)?,
                ring: rd.to_host_vec(s)?,
            })
        }
    }

    /// The host rule's KDA conv outputs for `tk` at `pos` over `ring`.
    fn kda_conv_host(l: &KdaLayer, tk: &KdaTokens, ring: &[f32], pos: &[u32], m: usize) -> ConvOut {
        kda_conv_prep_host(
            &tk.x, &tk.b, &tk.f, &l.w, &l.dt, &l.sa, pos, ring, KDA_SHAPE, KDA_LB, KDA_EPS, m,
        )
    }

    /// `m` tokens of KDA delta inputs from the host conv rule at the
    /// sequence start.
    fn kda_host_step(m: usize, seed: u64) -> Step {
        let l = KdaLayer::new(seed);
        let tk = KdaTokens::new(m, seed ^ 0x5eed);
        let zero = vec![0.0f32; KDA_SHAPE.ring_len()];
        Step::of(
            &kda_conv_host(&l, &tk, &zero, &positions(0, m), m),
            KDA_SHAPE,
        )
    }

    /// The host rule of one KDA delta call on the one-lane state `state`.
    fn kda_delta_ref(st: &Step, state: &[f32], m: usize) -> DeltaOut {
        kda_delta_host(&st.qkv, &st.beta, &st.decay, state, LANES, 0, KDA_SHAPE, m)
    }

    /// Clause 9 (KDA, GLM-5.3-Flash's shape): each entry against its host
    /// rule bit for bit, the shared parts against the GDN entries, the
    /// launchers' refusals.
    fn kda_clauses(cx: &Ctx<'_>) -> Result<Vec<bool>, GateError> {
        let unl = cx.gpu.unlabelled_sink();
        let nv = KDA_SHAPE.n_v;
        let l = KdaLayer::new(0x6b64_616c);
        let tk = KdaTokens::new(PROMPT, 0x6b74_6f6b);
        let dirty = Lcg(0x6b72).fill(KDA_SHAPE.ring_len(), -1.0, 1.0);
        let mut out = Vec::new();

        for m in [PROMPT, 1] {
            let t = tk.slice(0, m);
            let pos = positions(P0, m);
            let d = cx.kda_conv(unl, &l, &t, &dirty, &pos, KDA_LB, m)?;
            let h = kda_conv_host(&l, &t, &dirty, &pos, m);
            let (pass, line) = conv_same(&d, &h);
            println!(
                "kda conv m={m} at p0={P0} over a random ring: {line} {}",
                verdict(pass)
            );
            out.push(pass);
        }

        // The conv, the norm, β and the ring are gdn_conv_prep's: the same
        // channels, taps and β projection through both entries.
        let pos = positions(P0, PROMPT);
        let kd = cx.kda_conv(unl, &l, &tk, &dirty, &pos, KDA_LB, PROMPT)?;
        let gl = Layer {
            shape: KDA_SHAPE,
            eps: KDA_EPS,
            w: l.w.clone(),
            dt: l.dt[..nv].to_vec(),
            sa: l.sa.clone(),
            gain: l.gain.clone(),
        };
        let gt = Tokens {
            shape: KDA_SHAPE,
            x: tk.x.clone(),
            b: tk.b.clone(),
            a: Lcg(0x6b61).fill(PROMPT * nv, -3.0, 3.0),
        };
        let gd = cx.conv(unl, &gl, &gt, &dirty, &pos, PROMPT)?;
        let pass = bits_equal(&kd.y, &gd.y)
            && bits_equal(&kd.beta, &gd.beta)
            && bits_equal(&kd.ring, &gd.ring);
        println!(
            "kda conv m={PROMPT} vs gdn_conv_prep on the same channels, taps and b: y {} beta {} \
             ring {} {}",
            cmp(&kd.y, &gd.y),
            cmp(&kd.beta, &gd.beta),
            cmp(&kd.ring, &gd.ring),
            verdict(pass)
        );
        out.push(pass);

        let zero = vec![0.0f32; KDA_SHAPE.state_len()];
        let st = kda_host_step(PROMPT + 1, 0x6b64_6574);
        let prompt = st.slice(0, PROMPT);
        let d = cx.delta(unl, &prompt, &zero, 0, KDA_SHAPE, Gran::Key, PROMPT)?;
        let h = kda_delta_ref(&prompt, &zero, PROMPT);
        out.push(delta_line(
            &format!("kda prompt m={PROMPT} from a zero state"),
            &d,
            &h,
        ));
        let step = st.slice(PROMPT, 1);
        let ds = cx.delta(unl, &step, &h.state, 0, KDA_SHAPE, Gran::Key, 1)?;
        let hs = kda_delta_ref(&step, &h.state, 1);
        out.push(delta_line(
            &format!("kda decode step at depth {PROMPT}"),
            &ds,
            &hs,
        ));

        // kda_delta with every key given its head's decay is gdn_delta.
        let m = 64;
        let mut heads = st.slice(0, m);
        heads.decay = Lcg(0x6864).fill(m * nv, 0.5, 1.0);
        heads.dn = nv;
        let mut keys = st.slice(0, m);
        keys.decay = heads
            .decay
            .iter()
            .flat_map(|&v| std::iter::repeat_n(v, HEAD))
            .collect();
        let dh = cx.delta(unl, &heads, &zero, 0, KDA_SHAPE, Gran::Head, m)?;
        let dk = cx.delta(unl, &keys, &zero, 0, KDA_SHAPE, Gran::Key, m)?;
        out.push(delta_line(
            &format!("kda_delta m={m} with each key at its head's decay vs gdn_delta"),
            &dk,
            &dh,
        ));

        let mut r = Lcg(0x6b6e);
        let n = PROMPT * nv * HEAD;
        let o = r.fill(n, -0.5, 0.5);
        let z = r.fill(n, -4.0, 4.0);
        for m in [PROMPT, 1] {
            let len = m * nv * HEAD;
            let (o, z) = (&o[..len], &z[..len]);
            let d = cx.norm_act(unl, (o, z, &l.gain), KDA_EPS, nv, GATE_SIGMOID, m)?;
            let h = norm_gate_host::<GATE_SIGMOID>(o, z, &l.gain, KDA_EPS);
            let pass = bits_equal(&d, &h);
            println!(
                "kda norm_gate sigmoid n_v={nv} m={m} y {} {}",
                cmp(&d, &h),
                verdict(pass)
            );
            out.push(pass);
        }

        out.push(kda_depth(cx)?);
        out.push(kda_refusals(cx, &l, &tk, &st)?);
        Ok(out)
    }

    /// 4,096 tokens through the KDA conv and delta on the card in launches of
    /// 6, 1,018 and 3,072, the state after the last against the host rule
    /// over all 4,096 at once.
    fn kda_depth(cx: &Ctx<'_>) -> Result<bool, GateError> {
        let unl = cx.gpu.unlabelled_sink();
        let total = DEPTHS[DEPTHS.len() - 1];
        let l = KdaLayer::new(0x6b64_6570);
        let tk = KdaTokens::new(total, 0x6b68);
        let mut ring = vec![0.0f32; KDA_SHAPE.ring_len()];
        let mut state = vec![0.0f32; KDA_SHAPE.state_len()];
        let mut steps = Step::empty(KDA_SHAPE, KDA_SHAPE.n_v * HEAD);
        let mut t0 = 0;
        for &d in &DEPTHS {
            let m = d - t0;
            let c = cx.kda_conv(
                unl,
                &l,
                &tk.slice(t0, m),
                &ring,
                &positions(t0, m),
                KDA_LB,
                m,
            )?;
            ring = c.ring.clone();
            let st = Step::of(&c, KDA_SHAPE);
            state = cx
                .delta(unl, &st, &state, 0, KDA_SHAPE, Gran::Key, m)?
                .state;
            steps.extend(&st);
            t0 = d;
        }
        let zero = vec![0.0f32; KDA_SHAPE.state_len()];
        let h = kda_delta_ref(&steps, &zero, total);
        let pass = bits_equal(&state, &h.state);
        println!(
            "kda depth {total} tokens in launches of 6, 1018, 3072: state vs the host rule over \
             all {total} at once {} {}",
            cmp(&state, &h.state),
            verdict(pass)
        );
        Ok(pass)
    }

    /// The KDA launchers refuse by name a decay of the per-head length and
    /// a bound that is not below 0.
    fn kda_refusals(
        cx: &Ctx<'_>,
        l: &KdaLayer,
        tk: &KdaTokens,
        st: &Step,
    ) -> Result<bool, GateError> {
        let s = cx.stream();
        let unl = cx.gpu.unlabelled_sink();
        let nv = KDA_SHAPE.n_v;
        let one = st.slice(0, 1);
        let qkv = DeviceBuffer::from_host(s, &one.qkv)?;
        let beta = DeviceBuffer::from_host(s, &one.beta)?;
        let short = DeviceBuffer::from_host(s, &one.decay[..nv])?;
        let lw = DeviceBuffer::from_host(s, &[0u32])?;
        let mut o = DeviceBuffer::from_host(s, &vec![0.0f32; nv * HEAD])?;
        let mut state = DeviceBuffer::from_host(s, &vec![0.0f32; KDA_SHAPE.state_len()])?;
        let decay_said = match cx.k.delta.enqueue_kda_delta(
            s,
            DeltaArgs {
                qkv: &qkv,
                beta: &beta,
                decay: &short,
                lane: &lw,
                lane_at: 0,
                lanes: LANES,
                shape: KDA_SHAPE,
                m: 1,
                fault: unl,
                o: &mut o,
                state: &mut state,
            },
        ) {
            Ok(()) => (false, "launched".to_owned()),
            Err(e) => {
                let e = e.to_string();
                (e.contains(&format!("decay.len() {nv} < {}", nv * HEAD)), e)
            }
        };
        let lb_said = match cx.kda_conv(
            unl,
            l,
            &tk.slice(0, 1),
            &vec![0.0; KDA_SHAPE.ring_len()],
            &[0],
            0.0,
            1,
        ) {
            Ok(_) => (false, "launched".to_owned()),
            Err(e) => {
                let e = e.to_string();
                (e.contains("lb=0"), e)
            }
        };
        s.synchronize()?;
        let pass = decay_said.0 && lb_said.0;
        println!(
            "kda launchers refuse a per-head decay: \"{}\" and lb=0: \"{}\" {}",
            decay_said.1,
            lb_said.1,
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause 10 (K1, Qwen3.8's shape): the GDN conv and delta at 16/48
    /// tiled and the sigmoid-gated norm at n_v 48, each against its host
    /// rule bit for bit.
    fn k1_clauses(cx: &Ctx<'_>) -> Result<Vec<bool>, GateError> {
        let unl = cx.gpu.unlabelled_sink();
        let nv = K1_SHAPE.n_v;
        let l = Layer::with(0x006b_316c, K1_SHAPE);
        let tk = Tokens::with(PROMPT, 0x6b31_746b, K1_SHAPE);
        let dirty = Lcg(0x006b_3172).fill(K1_SHAPE.ring_len(), -1.0, 1.0);
        let mut out = Vec::new();

        let pos = positions(P0, PROMPT);
        let d = cx.conv(unl, &l, &tk, &dirty, &pos, PROMPT)?;
        let h = conv_host(&l, &tk, &dirty, &pos, PROMPT);
        let (pass, line) = conv_same(&d, &h);
        println!(
            "k1 conv n_k=16 n_v={nv} m={PROMPT} at p0={P0} over a random ring: {line} {}",
            verdict(pass)
        );
        out.push(pass);

        let zero_ring = vec![0.0f32; K1_SHAPE.ring_len()];
        let st = Step::of(
            &conv_host(&l, &tk, &zero_ring, &positions(0, PROMPT), PROMPT),
            K1_SHAPE,
        );
        let zero = vec![0.0f32; K1_SHAPE.state_len()];
        let d = cx.delta(unl, &st, &zero, 0, K1_SHAPE, Gran::Head, PROMPT)?;
        let h = delta_host(
            &st.qkv, &st.beta, &st.decay, &zero, LANES, 0, K1_SHAPE, PROMPT,
        );
        out.push(delta_line(
            &format!("k1 n_k=16 n_v={nv} tiled prompt m={PROMPT} from a zero state"),
            &d,
            &h,
        ));

        let mut r = Lcg(0x006b_316e);
        let n = PROMPT * nv * HEAD;
        let o = r.fill(n, -0.5, 0.5);
        let z = r.fill(n, -4.0, 4.0);
        for m in [PROMPT, 1] {
            let len = m * nv * HEAD;
            let (o, z) = (&o[..len], &z[..len]);
            let d = cx.norm_act(unl, (o, z, &l.gain), EPS, nv, GATE_SIGMOID, m)?;
            let h = norm_gate_host::<GATE_SIGMOID>(o, z, &l.gain, EPS);
            let pass = bits_equal(&d, &h);
            println!(
                "k1 norm_gate sigmoid n_v={nv} m={m} y {} {}",
                cmp(&d, &h),
                verdict(pass)
            );
            out.push(pass);
        }
        Ok(out)
    }

    /// Clause 11: the three new entries' fault sites, as clause 7.
    fn kda_fault_clauses(cx: &Ctx<'_>) -> Result<Vec<bool>, GateError> {
        let nv = KDA_SHAPE.n_v;
        let unl = cx.gpu.unlabelled_sink();
        let m = 16;
        let mut out = Vec::new();

        // KDA conv: a NaN forget value (token 3, head 5, key 40) reaches that
        // decay alone.
        let l = KdaLayer::new(0x006b_666c);
        let tk = KdaTokens::new(m, 0x006b_6674);
        let ring = Lcg(0x006b_6672).fill(KDA_SHAPE.ring_len(), -1.0, 1.0);
        let pos = positions(100, m);
        let clean = cx.kda_conv(unl, &l, &tk, &ring, &pos, KDA_LB, m)?;
        let layer = 30usize;
        let sink = cx.gpu.layer_sink(layer)?;
        let before = cx.gpu.fault()?;
        let mut bad_tk = tk.slice(0, m);
        let at = (3 * nv + 5) * HEAD + 40;
        bad_tk.f[at] = f32::NAN;
        let bad = cx.kda_conv(sink, &l, &bad_tk, &ring, &pos, KDA_LB, m)?;
        let word = cx.gpu.take_fault()?;
        let _ = cx.kda_conv(sink, &l, &tk, &ring, &pos, KDA_LB, m)?;
        let after = cx.gpu.fault()?;
        let outs = |c: &ConvOut| vec![c.y.clone(), c.beta.clone(), c.decay.clone(), c.ring.clone()];
        out.push(
            Planted {
                what: "kda conv NaN forget value",
                want: Fault::at(u32::try_from(layer)?, FaultSite::LinearConv),
                before,
                word,
                after,
                hit: vec![vec![], vec![], vec![at], vec![]],
                clean: outs(&clean),
                bad: outs(&bad),
            }
            .judge(),
        );

        // KDA delta: a NaN decay (token 9, head 4, key 17) reaches every
        // column of head 4 from token 9, and the head's whole state.
        let st = kda_host_step(m, 0x006b_6664);
        let zero = vec![0.0f32; KDA_SHAPE.state_len()];
        let clean = cx.delta(unl, &st, &zero, 0, KDA_SHAPE, Gran::Key, m)?;
        let layer = 31usize;
        let sink = cx.gpu.layer_sink(layer)?;
        let before = cx.gpu.fault()?;
        let mut bst = st.slice(0, m);
        let (h_nan, t_nan) = (4usize, 9usize);
        bst.decay[(t_nan * nv + h_nan) * HEAD + 17] = f32::NAN;
        let bad = cx.delta(sink, &bst, &zero, 0, KDA_SHAPE, Gran::Key, m)?;
        let word = cx.gpu.take_fault()?;
        let _ = cx.delta(sink, &st, &zero, 0, KDA_SHAPE, Gran::Key, m)?;
        let after = cx.gpu.fault()?;
        let ho: Vec<usize> = (t_nan..m)
            .flat_map(|t| (0..HEAD).map(move |c| (t * nv + h_nan) * HEAD + c))
            .collect();
        let hs: Vec<usize> = (h_nan * HEAD * HEAD..(h_nan + 1) * HEAD * HEAD).collect();
        out.push(
            Planted {
                what: "kda delta NaN key decay",
                want: Fault::at(u32::try_from(layer)?, FaultSite::LinearDelta),
                before,
                word,
                after,
                hit: vec![ho, hs],
                clean: vec![clean.o, clean.state],
                bad: vec![bad.o, bad.state],
            }
            .judge(),
        );

        // Sigmoid-gated norm: a NaN gate value (token 3, head 7, value 40).
        let mut r = Lcg(0x006b_666e);
        let n = m * nv * HEAD;
        let o = r.fill(n, -0.5, 0.5);
        let z = r.fill(n, -4.0, 4.0);
        let clean = cx.norm_act(unl, (&o, &z, &l.gain), KDA_EPS, nv, GATE_SIGMOID, m)?;
        let layer = 32usize;
        let sink = cx.gpu.layer_sink(layer)?;
        let before = cx.gpu.fault()?;
        let mut bz = z.clone();
        let at = (3 * nv + 7) * HEAD + 40;
        bz[at] = f32::NAN;
        let bad = cx.norm_act(sink, (&o, &bz, &l.gain), KDA_EPS, nv, GATE_SIGMOID, m)?;
        let word = cx.gpu.take_fault()?;
        let _ = cx.norm_act(sink, (&o, &z, &l.gain), KDA_EPS, nv, GATE_SIGMOID, m)?;
        let after = cx.gpu.fault()?;
        out.push(
            Planted {
                what: "norm_gate sigmoid NaN gate",
                want: Fault::at(u32::try_from(layer)?, FaultSite::LinearGate),
                before,
                word,
                after,
                hit: vec![vec![at]],
                clean: vec![clean],
                bad: vec![bad],
            }
            .judge(),
        );
        Ok(out)
    }

    /// Clause 12: `gdn_delta_lanes` against the one-lane rule.
    fn lanes_clauses(cx: &Ctx<'_>) -> Result<Vec<bool>, GateError> {
        let unl = cx.gpu.unlabelled_sink();
        let len = K1_SHAPE.state_len();
        let rule = |st: &Step, state: &[f32], m: usize| {
            delta_host(&st.qkv, &st.beta, &st.decay, state, LANES, 0, K1_SHAPE, m)
        };
        let l = Layer::with(0x006c_616e, K1_SHAPE);
        let tk = Tokens::with(PASS_ROWS, 0x6c61_6e74, K1_SHAPE);
        let zero_ring = vec![0.0f32; K1_SHAPE.ring_len()];
        let st = Step::of(
            &conv_host(&l, &tk, &zero_ring, &positions(0, PASS_ROWS), PASS_ROWS),
            K1_SHAPE,
        );
        let lanes = Lcg(0x6c61_6e73).fill(ROW_LANES * len, -0.1, 0.1);
        let (c, p0) = (2usize, 40u32);
        let stamps = [7, 9, p0, 11];
        let lane_of = |v: &[f32], i: usize| v[i * len..(i + 1) * len].to_vec();
        let mut out = Vec::new();

        for m in [PASS_ROWS, 1] {
            let t = st.slice(0, m);
            let call = LaneCall {
                lane: c as u32,
                p0,
                each: false,
            };
            let (d, sp) = cx.delta_lanes(unl, &t, (&lanes, &stamps), call, m)?;
            let h = rule(&t, &lane_of(&lanes, c), m);
            let lane_c = lane_of(&d.state, c);
            let others = (0..ROW_LANES)
                .filter(|&i| i != c)
                .all(|i| bits_equal(&lane_of(&d.state, i), &lane_of(&lanes, i)));
            let mut want = stamps;
            want[c] = p0 + m as u32;
            let pass =
                bits_equal(&d.o, &h.o) && bits_equal(&lane_c, &h.state) && others && sp == want;
            println!(
                "lanes final m={m} on lane {c} of {ROW_LANES}: o {} lane {c} {} other lanes untouched \
                 {others}, stamps {sp:?} (want {want:?}) {}",
                cmp(&d.o, &h.o),
                cmp(&lane_c, &h.state),
                verdict(pass)
            );
            out.push(pass);
        }

        for m in [ROW_LANES, ROW_LANES - 1] {
            let t = st.slice(0, m);
            let call = LaneCall {
                lane: c as u32,
                p0,
                each: true,
            };
            let (d, sp) = cx.delta_lanes(unl, &t, (&lanes, &stamps), call, m)?;
            let (mut state, mut o) = (lane_of(&lanes, c), Vec::new());
            let (mut rows_same, mut want_state, mut want) = (true, lanes.clone(), stamps);
            for j in 0..m {
                let h = rule(&st.slice(j, 1), &state, 1);
                let at = (c + j) % ROW_LANES;
                rows_same &= bits_equal(&lane_of(&d.state, at), &h.state);
                want_state[at * len..(at + 1) * len].copy_from_slice(&h.state);
                want[at] = p0 + j as u32 + 1;
                o.extend_from_slice(&h.o);
                state = h.state;
            }
            let whole = bits_equal(&d.state, &want_state);
            let pass = bits_equal(&d.o, &o) && rows_same && whole && sp == want;
            println!(
                "lanes rows m={m} from lane {c} of {ROW_LANES}: o vs {m} one-token calls {}; row j's \
                 state in lane (c + j) mod {ROW_LANES} {rows_same}, the whole state (a lane no row \
                 wrote untouched) {whole}; stamps {sp:?} (want {want:?}) {}",
                cmp(&d.o, &o),
                verdict(pass)
            );
            out.push(pass);
        }

        // Lane 2's stamp one past the call's first position: the word names
        // the layer and `delta_stamp`; every value is the clean launch's.
        let m = 1;
        let t = st.slice(0, m);
        let call = LaneCall {
            lane: c as u32,
            p0,
            each: false,
        };
        let (clean, _) = cx.delta_lanes(unl, &t, (&lanes, &stamps), call, m)?;
        let layer = 32usize;
        let sink = cx.gpu.layer_sink(layer)?;
        let before = cx.gpu.fault()?;
        let mut stale = stamps;
        stale[c] = p0 + 1;
        let (bad, _) = cx.delta_lanes(sink, &t, (&lanes, &stale), call, m)?;
        let word = cx.gpu.take_fault()?;
        let _ = cx.delta_lanes(sink, &t, (&lanes, &stamps), call, m)?;
        let after = cx.gpu.fault()?;
        out.push(
            Planted {
                what: "lanes stamp of lane 2 one past the call's position",
                want: Fault::at(u32::try_from(layer)?, FaultSite::DeltaStamp),
                before,
                word,
                after,
                hit: vec![vec![], vec![]],
                clean: vec![clean.o, clean.state],
                bad: vec![bad.o, bad.state],
            }
            .judge(),
        );

        // Row mode over more tokens than lanes is refused by name.
        let t = st.slice(0, ROW_LANES + 1);
        let call = LaneCall {
            lane: c as u32,
            p0,
            each: true,
        };
        let got = cx.delta_lanes(unl, &t, (&lanes, &stamps), call, ROW_LANES + 1);
        let named = matches!(&got, Err(e) if e.to_string().contains("row mode"));
        println!(
            "lanes launcher refuses row mode m={} over {ROW_LANES} lanes: \"{}\" {}",
            ROW_LANES + 1,
            match &got {
                Ok(_) => "launched".to_owned(),
                Err(e) => e.to_string(),
            },
            verdict(named)
        );
        out.push(named);
        Ok(out)
    }

    /// Clause 13: `kda_delta_lanes` against the one-lane rule, a verify of
    /// one token a row against two one-token calls chained.
    fn kda_lanes_clauses(cx: &Ctx<'_>) -> Result<Vec<bool>, GateError> {
        let unl = cx.gpu.unlabelled_sink();
        let len = KDA_SHAPE.state_len();
        let st = kda_host_step(PROMPT, 0x6b6c_616e);
        let lanes = Lcg(0x6b6c_6e73).fill(KDA_LANES * len, -0.1, 0.1);
        let (c, p0) = (1usize, 40u32);
        let stamps = [7, p0];
        let lane_of = |v: &[f32], i: usize| v[i * len..(i + 1) * len].to_vec();
        let mut out = Vec::new();
        let at_c = LaneCall {
            lane: c as u32,
            p0,
            each: false,
        };

        for m in [PROMPT, 1] {
            let t = st.slice(0, m);
            let (d, sp) = cx.kda_lanes(unl, &t, (&lanes, &stamps), (at_c, 0), m)?;
            let h = kda_delta_ref(&t, &lane_of(&lanes, c), m);
            let lane_c = lane_of(&d.state, c);
            let other = bits_equal(&lane_of(&d.state, 1 - c), &lane_of(&lanes, 1 - c));
            let mut want = stamps;
            want[c] = p0 + m as u32;
            let pass =
                bits_equal(&d.o, &h.o) && bits_equal(&lane_c, &h.state) && other && sp == want;
            println!(
                "kda lanes row 0 in place m={m} on lane {c} of {KDA_LANES}: o {} lane {c} {} the \
                 other lane untouched {other}, stamps {sp:?} (want {want:?}) {}",
                cmp(&d.o, &h.o),
                cmp(&lane_c, &h.state),
                verdict(pass)
            );
            out.push(pass);
        }

        // A verify of one token a row: row 0 in place, then row 1 at row
        // base 1 on what row 0 left, against two one-token calls chained.
        let (t0, t1) = (st.slice(0, 1), st.slice(1, 1));
        let (d0, sp0) = cx.kda_lanes(unl, &t0, (&lanes, &stamps), (at_c, 0), 1)?;
        let row1 = LaneCall {
            lane: c as u32,
            p0: p0 + 1,
            each: true,
        };
        let (d1, sp1) = cx.kda_lanes(unl, &t1, (&d0.state, &sp0), (row1, 1), 1)?;
        let h0 = kda_delta_ref(&t0, &lane_of(&lanes, c), 1);
        let h1 = kda_delta_ref(&t1, &h0.state, 1);
        let (w0, w1) = (c, (c + 1) % KDA_LANES);
        let rows = bits_equal(&d0.o, &h0.o)
            && bits_equal(&d1.o, &h1.o)
            && bits_equal(&lane_of(&d1.state, w0), &h0.state)
            && bits_equal(&lane_of(&d1.state, w1), &h1.state);
        let mut want = [0u32; KDA_LANES];
        want[w0] = p0 + 1;
        want[w1] = p0 + 2;
        let pass = rows && sp1 == want;
        println!(
            "kda lanes verify of 2 rows from lane {c}: row 0 in place, row 1 at base 1: o and row \
             j's state in lane (c + j) mod {KDA_LANES} vs two one-token calls chained {rows}; \
             stamps {sp1:?} (want {want:?}) {}",
            verdict(pass)
        );
        out.push(pass);

        // Row 1 on a lane 1 stamped one past its position: the word names
        // the layer and `delta_stamp`; every value is the clean launch's.
        let layer = 33usize;
        let (clean, _) = cx.kda_lanes(unl, &t1, (&d0.state, &sp0), (row1, 1), 1)?;
        let sink = cx.gpu.layer_sink(layer)?;
        let before = cx.gpu.fault()?;
        let mut stale = sp0.clone();
        stale[c] = p0 + 2;
        let (bad, _) = cx.kda_lanes(sink, &t1, (&d0.state, &stale), (row1, 1), 1)?;
        let word = cx.gpu.take_fault()?;
        let _ = cx.kda_lanes(sink, &t1, (&d0.state, &sp0), (row1, 1), 1)?;
        let after = cx.gpu.fault()?;
        out.push(
            Planted {
                what: "kda lanes stamp of lane 1 one past row 1's position",
                want: Fault::at(u32::try_from(layer)?, FaultSite::DeltaStamp),
                before,
                word,
                after,
                hit: vec![vec![], vec![]],
                clean: vec![clean.o, clean.state],
                bad: vec![bad.o, bad.state],
            }
            .judge(),
        );

        // The launcher's refusals, each by name before any launch.
        for (call, row, m, says) in [
            (row1, KDA_LANES, 1, "over 2 lanes"),
            (at_c, 1, 1, "only row 0"),
            (row1, 1, 2, "rows 1..3"),
        ] {
            let t = st.slice(0, m);
            let got = cx.kda_lanes(unl, &t, (&lanes, &stamps), (call, row), m);
            let named = matches!(&got, Err(e) if e.to_string().contains(says));
            println!(
                "kda lanes launcher refuses row {row} m={m} (row mode {}): \"{}\" {}",
                call.each,
                match &got {
                    Ok(_) => "launched".to_owned(),
                    Err(e) => e.to_string(),
                },
                verdict(named)
            );
            out.push(named);
        }
        Ok(out)
    }

    /// The node a set dumps as `name`, or when that row only relabels
    /// another (a permute, reshape, view or copy), the node it relabels,
    /// through its `src0`, with that row's `ne`.
    fn tap_or_src(man: &RefManifest, name: &str) -> Result<(Vec<f32>, [u64; 4]), GateError> {
        const RELABEL: [&str; 5] = ["PERMUTE", "RESHAPE", "VIEW", "CONT", "TRANSPOSE"];
        let (mut at, mut row) = man.tensor_at(name, 0)?;
        for _ in 0..4 {
            if !RELABEL.contains(&row.op.as_str()) {
                return Ok((ref_tensor_logical_in(&man.dir, row)?, row.ne));
            }
            (at, row) = man.last_before(at, row.src0.as_deref())?;
        }
        Err(format!("{name}: four relabelling rows and no node under them").into())
    }

    /// A set's tap `name` in its logical order.
    fn tap(man: &RefManifest, name: &str) -> Result<Vec<f32>, GateError> {
        Ok(ref_tensor_logical_in(&man.dir, man.tensor(name, 0)?)?)
    }

    /// A per-head tap of `t_n` tokens, `h_n` heads of `d_n` values, in our
    /// `[t][h][d]` order times `scale`: ik's is `[h][t][d]` (head-major, a
    /// batch's permuted norm) or `[t][h][d]`.
    fn heads_of(
        ik: &[f32],
        ne: [u64; 4],
        (t_n, h_n, d_n): (usize, usize, usize),
        scale: f32,
    ) -> Result<Vec<f32>, GateError> {
        let ne = [ne[0] as usize, ne[1] as usize, ne[2] as usize];
        let head_major = ne == [d_n, t_n, h_n] && t_n > 1;
        if !(head_major || ne == [d_n, h_n, t_n]) || ik.len() != t_n * h_n * d_n {
            return Err(format!("a head tap of ne {ne:?} for {t_n} x {h_n} x {d_n}").into());
        }
        let mut want = vec![0.0f32; t_n * h_n * d_n];
        for t in 0..t_n {
            for h in 0..h_n {
                for d in 0..d_n {
                    let src = if head_major {
                        (h * t_n + t) * d_n + d
                    } else {
                        (t * h_n + h) * d_n + d
                    };
                    want[(t * h_n + h) * d_n + d] = ik[src] * scale;
                }
            }
        }
        Ok(want)
    }

    /// `[h][a][b]` to `[h][b][a]` over `HEAD × HEAD` blocks: ik's state
    /// (`[k][v]` per head) to ours (`[v][k]`) and back.
    fn transpose_heads(x: &[f32]) -> Vec<f32> {
        let mut t = vec![0.0f32; x.len()];
        for (h, blk) in x.chunks(HEAD * HEAD).enumerate() {
            for a in 0..HEAD {
                for b in 0..HEAD {
                    t[h * HEAD * HEAD + b * HEAD + a] = blk[a * HEAD + b];
                }
            }
        }
        t
    }

    /// `‖ours − ik‖ / ‖ik‖` in f64; infinite on a length mismatch or a NaN.
    fn rel_l2(ours: &[f32], ik: &[f32]) -> f64 {
        if ours.len() != ik.len() {
            return f64::INFINITY;
        }
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (&a, &b) in ours.iter().zip(ik) {
            num += (f64::from(a) - f64::from(b)).powi(2);
            den += f64::from(b).powi(2);
        }
        let r = (num / den.max(f64::MIN_POSITIVE)).sqrt();
        if r.is_nan() { f64::INFINITY } else { r }
    }

    /// The batch set every KDA ik clause reads.
    const IK_KDA_SET: &str = "ref_glm5next";

    /// PIN(2026-09-28): the band of `kda_conv_prep` on ik's inputs, each
    /// output's `‖ours − ik‖ / ‖ik‖`. No recurrence carries an error from one
    /// token to the next; per value the known differences are the 4-tap
    /// conv's FMA chain against ik's products and sums (≤ 2 ulp), the q and
    /// k L2 norms' f64 sums in two orders (~√128·2⁻²⁴ ≈ 7e-7), our sigmoid
    /// and `expf_ik` against ik's graph ops, the decay's argument up to 5 in
    /// magnitude so its exp carries ≤ 5 × its rounding (≈ 6e-7): ~1e-6
    /// [derived]; 1e-5 is that with a 10× margin.
    const IK_BAND_PREP: f64 = 1e-5;

    /// `kda_conv_prep` for layer `l` on ik's own inputs of the batch set from
    /// a zero ring — the mixed q·k·v (`qkv_mixed`), β's raw projection
    /// (`beta_in`), the forget projection (`decay_raw`) and the file's conv
    /// taps, `dt_bias` and `ssm_a` — against ik: the v channels against
    /// `conv_output_silu`'s, q and k against `q_fused` and `k_fused` (ours ×
    /// `Q_SCALE` on q, as `ik_q` holds it), β against `sigmoid(beta_in)` and
    /// the decay against `expf_ik(g_in)`, each within [`IK_BAND_PREP`]. ik's
    /// `conv_states` of the set must be zero, as the set starts a sequence;
    /// another ring is refused by name.
    fn kda_prep_ik(
        cx: &Ctx<'_>,
        (man, split, l): (&RefManifest, &Split, usize),
        (ik_q, ik_k, ik_silu): (&[f32], &[f32], &[f32]),
        (ik_beta, ik_decay): (&[f32], &[f32]),
    ) -> Result<bool, GateError> {
        let (nk, nv) = (KDA_SHAPE.n_k, KDA_SHAPE.n_v);
        let ch = KDA_SHAPE.channels();
        let x = tap(man, &format!("qkv_mixed-{l}"))?;
        let t_n = x.len() / ch;
        let conv_states = tap(man, &format!("conv_states-{l}"))?;
        if conv_states.iter().any(|&v| v != 0.0) {
            return Err(format!(
                "layer {l}: {IK_KDA_SET}'s conv_states are not zero; the prep clause starts a \
                 sequence from a zero ring"
            )
            .into());
        }
        let tk = KdaTokens {
            x,
            b: tap(man, &format!("beta_in-{l}"))?,
            f: tap(man, &format!("decay_raw-{l}"))?,
        };
        let mut w = Vec::with_capacity(CONV_TAPS * ch);
        for part in ['q', 'k', 'v'] {
            let name = format!("blk.{l}.ssm_conv1d_{part}.weight");
            w.extend(split_f32(split, &name, CONV_TAPS * nk * HEAD)?);
        }
        let layer = KdaLayer {
            w,
            dt: split_f32(split, &format!("blk.{l}.ssm_dt.bias"), nv * HEAD)?,
            sa: split_f32(split, &format!("blk.{l}.ssm_a"), nv)?,
            gain: Vec::new(),
        };
        if tk.x.len() != t_n * ch || tk.b.len() != t_n * nv || tk.f.len() != t_n * nv * HEAD {
            return Err(format!(
                "layer {l}: qkv_mixed {}, beta_in {} and decay_raw {} values, want {t_n} x {ch}, \
                 {nv} and {}",
                tk.x.len(),
                tk.b.len(),
                tk.f.len(),
                nv * HEAD
            )
            .into());
        }
        let zero = vec![0.0f32; KDA_SHAPE.ring_len()];
        let pos = positions(0, t_n);
        let d = cx.kda_conv(
            cx.gpu.unlabelled_sink(),
            &layer,
            &tk,
            &zero,
            &pos,
            KDA_LB,
            t_n,
        )?;
        let part = |y: &[f32], from: usize, to: usize| -> Vec<f32> {
            (0..t_n)
                .flat_map(|t| y[t * ch + from..t * ch + to].to_vec())
                .collect()
        };
        let q_end = nk * HEAD;
        let k_end = 2 * nk * HEAD;
        let rels = [
            ("q", rel_l2(&part(&d.y, 0, q_end), ik_q)),
            ("k", rel_l2(&part(&d.y, q_end, k_end), ik_k)),
            (
                "v",
                rel_l2(&part(&d.y, k_end, ch), &part(ik_silu, k_end, ch)),
            ),
            ("beta", rel_l2(&d.beta, ik_beta)),
            ("decay", rel_l2(&d.decay, ik_decay)),
        ];
        let ok = rels.iter().all(|&(_, e)| e <= IK_BAND_PREP);
        println!(
            "kda ik prep layer {l} ({t_n} tokens of {IK_KDA_SET}): {} (band {IK_BAND_PREP:.0e}) {}",
            rels.iter()
                .map(|(n, e)| format!("{n} rel={e:.3e}"))
                .collect::<Vec<_>>()
                .join(" "),
            verdict(ok)
        );
        Ok(ok)
    }

    /// The `kda_ik` case (KDA against ik): per KDA layer of the `ref_glm5next` set,
    /// `kda_conv_prep` on ik's own inputs ([`kda_prep_ik`]), `kda_delta` on ik's own inputs (`q_fused` ×[`Q_SCALE`], `k_fused`, the
    /// v channels of `conv_output_silu`, `sigmoid(beta_in)`, `expf_ik(g_in)`,
    /// `state_in` transposed) against `attn_output` and `new_state`
    /// (transposed) within [`IK_BAND_DELTA`], and the sigmoid-gated norm on
    /// ik's `attn_output` and `z` with the file's `ssm_norm` against
    /// `final_output` within [`IK_BAND_NORM`]. Three known differences,
    /// named, not banded away: ik sums `Σ_col S·(k·d)` in order where we
    /// round `d·S` and split the dot over lanes; ik's β and decay take libm
    /// `expf`, ours [`expf_ik`]; ik clamps the state to ±1e6 every step, ours
    /// raises the fault word instead — an ik state value at ±1e6 fails the
    /// clause by name. The case `kda_ik`, run only when named: without the
    /// glm5next family, or with it and a set, tap or tensor missing, it is an
    /// error by name, never a pass.
    fn kda_ik_band(cx: &Ctx<'_>) -> Result<bool, GateError> {
        let fam = refset::arch::node_dumps("glm5next").ok_or(
            "kda ik band (attn_output, new_state, final_output against ref_glm5next): \
             glm5next refset family not in this tree",
        )?;
        let set = fam
            .sets
            .iter()
            .find(|s| **s == IK_KDA_SET)
            .ok_or_else(|| format!("family {} has no set {IK_KDA_SET}", fam.name))?;
        let man = RefManifest::open(&fam.path(set), fam)?;
        let model = fam.runs()?;
        let split = Split::open(&model).map_err(|e| format!("open {model}: {e}"))?;
        let n_layer = split
            .arch_get_u64("block_count")
            .ok_or_else(|| format!("{model}: no block_count"))? as usize;
        let (nk, nv) = (KDA_SHAPE.n_k, KDA_SHAPE.n_v);
        let ch = KDA_SHAPE.channels();
        let unl = cx.gpu.unlabelled_sink();
        let mut pass = true;
        let mut layers = 0usize;
        for l in 0..n_layer {
            if man.tensor(&format!("new_state-{l}"), 0).is_err() {
                continue;
            }
            layers += 1;
            let (q, q_ne) = tap_or_src(&man, &format!("q_fused-{l}"))?;
            let t_n = q.len() / (nk * HEAD);
            let q = heads_of(&q, q_ne, (t_n, nk, HEAD), Q_SCALE)?;
            let (k, k_ne) = tap_or_src(&man, &format!("k_fused-{l}"))?;
            let k = heads_of(&k, k_ne, (t_n, nk, HEAD), 1.0)?;
            let silu = tap(&man, &format!("conv_output_silu-{l}"))?;
            if silu.len() != t_n * ch {
                return Err(format!(
                    "layer {l}: conv_output_silu holds {} values, want {t_n} x {ch}",
                    silu.len()
                )
                .into());
            }
            let mut qkv = vec![0.0f32; t_n * ch];
            for t in 0..t_n {
                let row = &mut qkv[t * ch..(t + 1) * ch];
                row[..nk * HEAD].copy_from_slice(&q[t * nk * HEAD..(t + 1) * nk * HEAD]);
                row[nk * HEAD..2 * nk * HEAD]
                    .copy_from_slice(&k[t * nk * HEAD..(t + 1) * nk * HEAD]);
                row[2 * nk * HEAD..].copy_from_slice(&silu[t * ch + 2 * nk * HEAD..(t + 1) * ch]);
            }
            let beta: Vec<f32> = tap(&man, &format!("beta_in-{l}"))?
                .iter()
                .map(|&b| sigmoid(b))
                .collect();
            let decay: Vec<f32> = tap(&man, &format!("g_in-{l}"))?
                .iter()
                .map(|&g| expf_ik(g))
                .collect();
            let state_in = tap(&man, &format!("state_in-{l}"))?;
            if beta.len() != t_n * nv
                || decay.len() != t_n * nv * HEAD
                || state_in.len() != KDA_SHAPE.state_len()
            {
                return Err(format!(
                    "layer {l}: beta_in {}, g_in {} and state_in {} values, want {}, {} and {}",
                    beta.len(),
                    decay.len(),
                    state_in.len(),
                    t_n * nv,
                    t_n * nv * HEAD,
                    KDA_SHAPE.state_len()
                )
                .into());
            }
            let prep_ok = kda_prep_ik(cx, (&man, &split, l), (&q, &k, &silu), (&beta, &decay))?;
            let st = Step {
                qkv,
                beta,
                decay,
                ch,
                nv,
                dn: nv * HEAD,
            };
            let d = cx.delta(
                unl,
                &st,
                &transpose_heads(&state_in),
                0,
                KDA_SHAPE,
                Gran::Key,
                t_n,
            )?;
            let ik_o = tap(&man, &format!("attn_output-{l}"))?;
            let ik_state = tap(&man, &format!("new_state-{l}"))?;
            let clamped = ik_state.iter().filter(|v| v.abs() >= 1e6).count();
            let e_o = rel_l2(&d.o, &ik_o);
            let e_s = rel_l2(&d.state, &transpose_heads(&ik_state));
            let z = tap(&man, &format!("z-{l}"))?;
            let gain = split_f32(&split, &format!("blk.{l}.ssm_norm.weight"), HEAD)?;
            let y = cx.norm_act(unl, (&ik_o, &z, &gain), KDA_EPS, nv, GATE_SIGMOID, t_n)?;
            let e_y = rel_l2(&y, &tap(&man, &format!("final_output-{l}"))?);
            let ok = prep_ok
                && clamped == 0
                && e_o <= IK_BAND_DELTA
                && e_s <= IK_BAND_DELTA
                && e_y <= IK_BAND_NORM;
            pass &= ok;
            println!(
                "kda ik band layer {l} ({t_n} tokens of {IK_KDA_SET}): attn_output rel={e_o:.3e} \
                 new_state rel={e_s:.3e} (band {IK_BAND_DELTA:.0e}) final_output rel={e_y:.3e} \
                 (band {IK_BAND_NORM:.0e}) ik state values at the ±1e6 clamp {clamped} {}",
                verdict(ok)
            );
        }
        if layers == 0 {
            return Err(format!("kda ik band: {IK_KDA_SET} holds no new_state row").into());
        }
        Ok(pass)
    }
}
