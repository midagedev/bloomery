//! GPU gate for a PLE site's two kernels (`bloomery_gpu::ple`), with no model:
//! synthetic streams, keys, values, gains and conv taps at Qwen3.8-Flash-Next's
//! shape (4 streams of 2,560, 10,240 conv channels, 4 taps 3 apart) from fixed
//! seeds. Every clause draws its inputs from the host, so a defect turns only
//! its own kernel's lines red.
//!
//! Clauses:
//! 1. `gate`: `ple_gate` at m = 1 and 5 against its host rule (`ple::gate_host`)
//!    bit for bit — `gv`, `ngv` and the gate — a rerun bit-identical, and every
//!    value within its derived bound of the exact f64 computation
//!    ([`exact_gate`], [`gate_bounds`]).
//! 2. `conv`: `ple_conv` of 4,096 (a whole ubatch in one launch), 64, 9, 2 and
//!    1 tokens at position 1,000 over a random ring against its host rule
//!    (`ple::conv_host`) bit for bit, the output and the ring after it.
//! 3. `mcol`: the gate at m = 8 and 40 is the m = 1 launches of its columns
//!    bit for bit; the conv of 8 tokens from position 500 and of 40 from 1,000
//!    (slots wrapping the ring twice, the first nine rows reading it) is the
//!    one-token launches chained through the ring, output and ring.
//! 4. `wrap`: 41 tokens from position 0 in one launch and in 41 one-token
//!    launches chained through the ring, both the host rule's: positions 8, 9,
//!    10 (the first tap to reach position 0) and 16, 17, 18, 33, 34, 35 (the
//!    ring's slots wrapping) are among them and are printed.
//! 5. `rollback`: 28 tokens from position 0 in launches of 20 and 8, then a
//!    rollback to position 23 — four new tokens launched on the ring as the
//!    pass left it — equal to the host rule over the 23 old and 4 new tokens
//!    from a zero ring.
//! 6. `reset`: `reset_ring` leaves the ring all zero; tokens from position 0
//!    over a random ring give the zero ring's output.
//! 7. `fault`: a NaN key value, a key row of 1e30 (finite values whose squares
//!    overflow), an infinite value (the gated value's norm), a NaN in the conv
//!    input and a position out of line, each with a layer's sink: the word
//!    names that layer and `ple`, the values the planted input reaches are
//!    NaN, every other value is the clean launch's bit for bit, and the word
//!    is clean before and after.
//! 8. `refuse`: the launchers refuse, by name, a conv of another tap count or
//!    dilation, no stream, no token, a ring or key shorter than the call's.
//! 9. `shape`: both entries compile with no local depot.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_ple: built without the `gpu` feature; see `just gate-gpu-ple`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_ple", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::ple::{
        CLAMP_MIN, ConvOut, DILATION, GateOut, PleConvArgs, PleGateArgs, PleKernels, RING_ROWS,
        ROW, TAPS, conv_host, gate_host, ring_len,
    };
    use bloomery_gpu::{Fault, FaultSink, FaultSite, Gpu};
    use bloomery_gpu_gates::{GateError, bits_equal, checks_failed, no_local_depot, verdict};
    use cuda_core::DeviceBuffer;

    /// Streams: Qwen3.8's `hyper_connection.count`.
    const HC: usize = 4;
    /// Conv channels, `HC·ROW`.
    const CH: usize = HC * ROW;
    /// The file's `attention.layer_norm_rms_epsilon`.
    const EPS: f32 = 1e-6;
    /// f32's unit roundoff.
    const U: f64 = 1.0 / (1u64 << 24) as f64;

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
        fn fill(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
            (0..n).map(|_| lo + (hi - lo) * self.unit()).collect()
        }
    }

    /// The gate's inputs for `m` tokens: keys and streams in [−2, 2), values
    /// in [−1, 1), gains in [0.5, 1.5).
    #[derive(Clone)]
    struct GateIn {
        key: Vec<f32>,
        value: Vec<f32>,
        x: Vec<f32>,
        gk: Vec<f32>,
        gq: Vec<f32>,
        gc: Vec<f32>,
        m: usize,
    }

    impl GateIn {
        fn new(m: usize, seed: u64) -> GateIn {
            let mut r = Lcg(seed);
            GateIn {
                key: r.fill(m * CH, -2.0, 2.0),
                value: r.fill(m * ROW, -1.0, 1.0),
                x: r.fill(m * CH, -2.0, 2.0),
                gk: r.fill(CH, 0.5, 1.5),
                gq: r.fill(CH, 0.5, 1.5),
                gc: r.fill(CH, 0.5, 1.5),
                m,
            }
        }

        /// Token `t` alone.
        fn column(&self, t: usize) -> GateIn {
            GateIn {
                key: self.key[t * CH..(t + 1) * CH].to_vec(),
                value: self.value[t * ROW..(t + 1) * ROW].to_vec(),
                x: self.x[t * CH..(t + 1) * CH].to_vec(),
                m: 1,
                ..self.clone()
            }
        }

        fn host(&self) -> GateOut {
            gate_host(
                &self.key,
                &self.value,
                &self.x,
                &self.gk,
                &self.gq,
                &self.gc,
                EPS,
                HC,
                self.m,
            )
        }
    }

    /// The conv's inputs for `m` tokens: `ngv`, `gv` and the streams in
    /// [−2, 2); the taps in [−0.5, 0.5).
    #[derive(Clone)]
    struct ConvIn {
        ngv: Vec<f32>,
        gv: Vec<f32>,
        x: Vec<f32>,
        w: Vec<f32>,
        m: usize,
    }

    impl ConvIn {
        fn new(m: usize, seed: u64) -> ConvIn {
            let mut r = Lcg(seed);
            ConvIn {
                ngv: r.fill(m * CH, -2.0, 2.0),
                gv: r.fill(m * CH, -2.0, 2.0),
                x: r.fill(m * CH, -2.0, 2.0),
                w: r.fill(TAPS * CH, -0.5, 0.5),
                m,
            }
        }

        /// Tokens `a .. b`.
        fn rows(&self, a: usize, b: usize) -> ConvIn {
            ConvIn {
                ngv: self.ngv[a * CH..b * CH].to_vec(),
                gv: self.gv[a * CH..b * CH].to_vec(),
                x: self.x[a * CH..b * CH].to_vec(),
                w: self.w.clone(),
                m: b - a,
            }
        }

        /// Tokens `a .. b` of `self`, then every token of `tail`: one layer's
        /// tokens, so `tail` carries `self`'s taps (panics by name otherwise —
        /// a reference over two layers' taps is not the rule of either).
        fn then(&self, b: usize, tail: &ConvIn) -> ConvIn {
            assert!(
                bits_equal(&self.w, &tail.w),
                "ConvIn::then: the tail's taps are not this layer's"
            );
            let join = |x: &[f32], y: &[f32]| [&x[..b * CH], y].concat();
            ConvIn {
                ngv: join(&self.ngv, &tail.ngv),
                gv: join(&self.gv, &tail.gv),
                x: join(&self.x, &tail.x),
                w: self.w.clone(),
                m: b + tail.m,
            }
        }

        fn host(&self, pos: &[u32], ring: &[f32]) -> ConvOut {
            conv_host(&self.ngv, &self.gv, &self.x, &self.w, pos, ring, HC, self.m)
        }
    }

    fn positions(p0: u32, m: usize) -> Vec<u32> {
        (0..m as u32).map(|t| p0 + t).collect()
    }

    struct Cx<'a> {
        gpu: &'a Gpu,
        k: &'a PleKernels,
    }

    impl Cx<'_> {
        fn gate(&self, fault: FaultSink, i: &GateIn) -> Result<GateOut, GateError> {
            let s = self.gpu.stream();
            let up = |v: &[f32]| DeviceBuffer::from_host(s, v);
            let (key, value, x) = (up(&i.key)?, up(&i.value)?, up(&i.x)?);
            let (gk, gq, gc) = (up(&i.gk)?, up(&i.gq)?, up(&i.gc)?);
            let mut gv = DeviceBuffer::<f32>::zeroed(s, i.m * CH)?;
            let mut ngv = DeviceBuffer::<f32>::zeroed(s, i.m * CH)?;
            let mut gate = DeviceBuffer::<f32>::zeroed(s, i.m * HC)?;
            self.k.enqueue_gate(
                s,
                PleGateArgs {
                    key: &key,
                    value: &value,
                    x: &x,
                    gain_key: &gk,
                    gain_query: &gq,
                    gain_conv: &gc,
                    eps: EPS,
                    hc: HC,
                    m: i.m,
                    fault,
                    gv: &mut gv,
                    ngv: &mut ngv,
                    gate: &mut gate,
                },
            )?;
            s.synchronize()?;
            Ok(GateOut {
                gv: gv.to_host_vec(s)?,
                ngv: ngv.to_host_vec(s)?,
                gate: gate.to_host_vec(s)?,
            })
        }

        /// One conv launch of `i` at `pos` over a ring holding `ring`; the
        /// output and the ring after it.
        fn conv(
            &self,
            fault: FaultSink,
            i: &ConvIn,
            pos: &[u32],
            ring: &[f32],
        ) -> Result<ConvOut, GateError> {
            let s = self.gpu.stream();
            let up = |v: &[f32]| DeviceBuffer::from_host(s, v);
            let (ngv, gv, x, w) = (up(&i.ngv)?, up(&i.gv)?, up(&i.x)?, up(&i.w)?);
            let pd = DeviceBuffer::from_host(s, pos)?;
            let mut out = DeviceBuffer::<f32>::zeroed(s, i.m * CH)?;
            let mut rd = DeviceBuffer::from_host(s, ring)?;
            self.k.enqueue_conv(
                s,
                PleConvArgs {
                    ngv: &ngv,
                    gv: &gv,
                    x: &x,
                    w: &w,
                    pos: &pd,
                    taps: TAPS,
                    dilation: DILATION,
                    hc: HC,
                    m: i.m,
                    fault,
                    out: &mut out,
                    ring: &mut rd,
                },
            )?;
            s.synchronize()?;
            Ok(ConvOut {
                out: out.to_host_vec(s)?,
                ring: rd.to_host_vec(s)?,
            })
        }

        /// `i` launched one token at a time from position `p0`, the ring
        /// carried between launches; the outputs in token order and the ring
        /// after the last.
        fn conv_chained(&self, i: &ConvIn, p0: u32, ring: &[f32]) -> Result<ConvOut, GateError> {
            let unl = self.gpu.unlabelled_sink();
            let mut ring = ring.to_vec();
            let mut out = Vec::with_capacity(i.m * CH);
            for t in 0..i.m {
                let r = self.conv(unl, &i.rows(t, t + 1), &[p0 + t as u32], &ring)?;
                out.extend(r.out);
                ring = r.ring;
            }
            Ok(ConvOut { out, ring })
        }
    }

    pub fn run() -> Result<(), GateError> {
        if std::env::args().len() > 1 {
            return Err("usage: gate_ple".into());
        }
        let gpu = Gpu::new()?;
        let k = PleKernels::load(gpu.context())?;
        let cx = Cx { gpu: &gpu, k: &k };
        println!(
            "gate_ple: device {} — hc={HC} row={ROW} channels={CH} taps={TAPS} dilation={DILATION} \
             ring={RING_ROWS} rows, synthetic",
            gpu.device_name()?
        );
        let mut failed = 0u32;
        let mut clauses = 0u32;
        let mut tally = |pass: bool| {
            clauses += 1;
            failed += u32::from(!pass);
        };
        for pass in gate_clauses(&cx)? {
            tally(pass);
        }
        for pass in conv_clauses(&cx)? {
            tally(pass);
        }
        for pass in mcol_clauses(&cx)? {
            tally(pass);
        }
        tally(wrap(&cx)?);
        tally(rollback(&cx)?);
        tally(reset(&cx)?);
        for pass in fault_clauses(&cx)? {
            tally(pass);
        }
        tally(refuse(&cx)?);
        tally(no_local_depot(&["ple_gate", "ple_conv"])?);
        let pass = failed == 0;
        println!(
            "gate_ple: {clauses} clauses, {failed} failed — {}",
            verdict(pass)
        );
        if !pass {
            return Err(checks_failed());
        }
        Ok(())
    }

    // ------------------------------------------------------------- 1. gate

    /// The exact gate in f64 on the same inputs: `kn = k/√(Σk²/n + ε)·γk`,
    /// `qn` alike, `dot = Σ kn·qn`, `s = dot/√n`, `m = sgn(s)·√max(|s|,
    /// 1e-6)`, `g = σ(m)`, `gv = v·g`, `ngv = gv/√(Σgv²/n + ε)·γc`; and per
    /// stream `s`, `Σ|kn·qn|` and the mean square of `gv`.
    struct Exact {
        gv: Vec<f64>,
        ngv: Vec<f64>,
        gate: Vec<f64>,
        s: Vec<f64>,
        dot: Vec<f64>,
        abs_sum: Vec<f64>,
        gv_ms: Vec<f64>,
    }

    fn exact_gate(i: &GateIn) -> Exact {
        let n = ROW as f64;
        let eps = f64::from(EPS);
        let norm = |r: &[f32]| -> f64 {
            let ms = r.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / n;
            1.0 / (ms + eps).sqrt()
        };
        let blocks = i.m * HC;
        let mut e = Exact {
            gv: vec![0.0; blocks * ROW],
            ngv: vec![0.0; blocks * ROW],
            gate: vec![0.0; blocks],
            s: vec![0.0; blocks],
            dot: vec![0.0; blocks],
            abs_sum: vec![0.0; blocks],
            gv_ms: vec![0.0; blocks],
        };
        for b in 0..blocks {
            let (t, st) = (b / HC, b % HC);
            let kr = &i.key[b * ROW..][..ROW];
            let xr = &i.x[b * ROW..][..ROW];
            let vr = &i.value[t * ROW..][..ROW];
            let g = |v: &[f32], d: usize| f64::from(v[st * ROW + d]);
            let (sk, sx) = (norm(kr), norm(xr));
            let (mut dot, mut abs) = (0.0f64, 0.0f64);
            for d in 0..ROW {
                let p =
                    (f64::from(kr[d]) * sk * g(&i.gk, d)) * (f64::from(xr[d]) * sx * g(&i.gq, d));
                dot += p;
                abs += p.abs();
            }
            let s = dot / n.sqrt();
            let m = s.signum() * s.abs().max(f64::from(CLAMP_MIN)).sqrt();
            let gate = 1.0 / (1.0 + (-m).exp());
            let gv: Vec<f64> = vr.iter().map(|&v| f64::from(v) * gate).collect();
            let ms = gv.iter().map(|v| v * v).sum::<f64>() / n;
            let sc = 1.0 / (ms + eps).sqrt();
            for d in 0..ROW {
                e.gv[b * ROW + d] = gv[d];
                e.ngv[b * ROW + d] = gv[d] * sc * g(&i.gc, d);
            }
            (e.gate[b], e.s[b], e.dot[b], e.abs_sum[b], e.gv_ms[b]) = (gate, s, dot, abs, ms);
        }
        e
    }

    /// Per value, how far the module's f32 rule may sit from [`Exact`]
    /// [derived, first order in u = 2⁻²⁴; PIN(2026-09-27): judged as is, no
    /// factor on top]:
    /// - a row's sum of squares: ten fused multiply-adds, five butterfly
    ///   levels, three tree levels over positive terms, ≤ 18u; `rms_scale`
    ///   (divide, add ε, sqrt, reciprocal) takes half of that plus 3u, ≤ 12u,
    ///   taken as 13u;
    /// - `kn`, `qn`: the scale and two roundings, 15u each; a product `p` one
    ///   more: 31u·|p|; the f64 sum adds nothing at this order;
    /// - `s`: `|Δs| ≤ (31u·Σ|p| + u·|dot|)/√n + 3u·|s|` (the dot's f32
    ///   rounding, `1/√n`'s own and the product's);
    /// - `m`: `|Δs|/(2√(|s| − |Δs|)) + u·|m|` while the clamp and sign cannot
    ///   move, else `2√(|s| + |Δs| + 1e-6)`;
    /// - gate: the sigmoid's slope at the point of `[m − Δm, m + Δm]` nearest
    ///   0 times `Δm`, plus 4u·g (the exponential rounded once, the add, the
    ///   division);
    /// - `gv = v·g`: `|v|·Δg + u·|gv|`;
    /// - `ngv`: a relative error common to a row of `gv` cancels in its norm
    ///   except for the share ε takes of the mean square, `(Δg/g)·ε/(ms + ε)`;
    ///   what remains is the norm's 13u, one rounding of `gv` and the two of
    ///   `(v·scale)·gain`: 16u·|ngv| plus that share.
    struct Bounds {
        gv: Vec<f64>,
        ngv: Vec<f64>,
        gate: Vec<f64>,
    }

    fn gate_bounds(i: &GateIn, e: &Exact) -> Bounds {
        let n = ROW as f64;
        let sig = |m: f64| 1.0 / (1.0 + (-m).exp());
        let mut b = Bounds {
            gv: vec![0.0; e.gv.len()],
            ngv: vec![0.0; e.ngv.len()],
            gate: vec![0.0; e.gate.len()],
        };
        for k in 0..e.gate.len() {
            let t = k / HC;
            let s = e.s[k];
            let ds = (31.0 * U * e.abs_sum[k] + U * e.dot[k].abs()) / n.sqrt() + 3.0 * U * s.abs();
            let m = s.signum() * s.abs().max(f64::from(CLAMP_MIN)).sqrt();
            let dm = if s.abs() - ds > f64::from(CLAMP_MIN) {
                ds / (2.0 * (s.abs() - ds).sqrt()) + U * m.abs()
            } else {
                2.0 * (s.abs() + ds + f64::from(CLAMP_MIN)).sqrt()
            };
            let near0 = if m - dm > 0.0 {
                m - dm
            } else if m + dm < 0.0 {
                m + dm
            } else {
                0.0
            };
            let slope = sig(near0) * (1.0 - sig(near0));
            let g = e.gate[k];
            let dg = slope * dm + 4.0 * U * g;
            b.gate[k] = dg;
            let share = (dg / g) * f64::from(EPS) / (e.gv_ms[k] + f64::from(EPS));
            for d in 0..ROW {
                let at = k * ROW + d;
                let v = f64::from(i.value[t * ROW + d]);
                b.gv[at] = v.abs() * dg + U * e.gv[at].abs();
                b.ngv[at] = (16.0 * U + share) * e.ngv[at].abs();
            }
        }
        b
    }

    /// How many values of `got` sit farther than their bound from `want`, and
    /// the largest distance over bound.
    fn over(got: &[f32], want: &[f64], bound: &[f64]) -> (usize, f64) {
        got.iter()
            .zip(want)
            .zip(bound)
            .fold((0, 0.0f64), |(n, worst), ((&g, &w), &bd)| {
                let d = (f64::from(g) - w).abs();
                let r = if bd > 0.0 { d / bd } else { f64::INFINITY };
                (n + usize::from(d.is_nan() || d > bd), worst.max(r))
            })
    }

    fn gate_same(a: &GateOut, b: &GateOut) -> bool {
        bits_equal(&a.gv, &b.gv) && bits_equal(&a.ngv, &b.ngv) && bits_equal(&a.gate, &b.gate)
    }

    /// Clause 1.
    fn gate_clauses(cx: &Cx<'_>) -> Result<Vec<bool>, GateError> {
        let unl = cx.gpu.unlabelled_sink();
        let mut out = Vec::new();
        for (m, seed) in [(1usize, 0x6731u64), (5, 0x6735)] {
            let i = GateIn::new(m, seed);
            let got = cx.gate(unl, &i)?;
            let again = cx.gate(unl, &i)?;
            let host = i.host();
            let e = exact_gate(&i);
            let bd = gate_bounds(&i, &e);
            let (og, wg) = over(&got.gate, &e.gate, &bd.gate);
            let (ov, wv) = over(&got.gv, &e.gv, &bd.gv);
            let (on, wn) = over(&got.ngv, &e.ngv, &bd.ngv);
            let exact = gate_same(&got, &host);
            let rerun = gate_same(&got, &again);
            let pass = exact && rerun && og + ov + on == 0;
            println!(
                "gate m={m}: bit_exact_host={exact} bit_identical_rerun={rerun} f64 over-bound \
                 gate={og} (worst {wg:.2} of bound) gv={ov} ({wv:.2}) ngv={on} ({wn:.2}) \
                 gates={:?} {}",
                got.gate,
                verdict(pass)
            );
            out.push(pass);
        }
        Ok(out)
    }

    // ------------------------------------------------------------- 2. conv

    fn conv_same(a: &ConvOut, b: &ConvOut) -> bool {
        bits_equal(&a.out, &b.out) && bits_equal(&a.ring, &b.ring)
    }

    /// Clause 2.
    fn conv_clauses(cx: &Cx<'_>) -> Result<Vec<bool>, GateError> {
        let unl = cx.gpu.unlabelled_sink();
        let ring = Lcg(0x7269_6e67).fill(ring_len(HC), -2.0, 2.0);
        let mut out = Vec::new();
        for m in [4096usize, 64, 9, 2, 1] {
            let i = ConvIn::new(m, 0x636f + m as u64);
            let pos = positions(1_000, m);
            let got = cx.conv(unl, &i, &pos, &ring)?;
            let pass = conv_same(&got, &i.host(&pos, &ring));
            println!(
                "conv m={m} at 1000 over a random ring: bit_exact_host out+ring {}",
                verdict(pass)
            );
            out.push(pass);
        }
        Ok(out)
    }

    // ------------------------------------------------------------- 3. mcol

    /// Clause 3.
    fn mcol_clauses(cx: &Cx<'_>) -> Result<Vec<bool>, GateError> {
        let unl = cx.gpu.unlabelled_sink();
        let mut out = Vec::new();
        for m in [8usize, 40] {
            let i = GateIn::new(m, 0x6d63 + m as u64);
            let all = cx.gate(unl, &i)?;
            let mut cols = GateOut {
                gv: Vec::new(),
                ngv: Vec::new(),
                gate: Vec::new(),
            };
            for t in 0..m {
                let c = cx.gate(unl, &i.column(t))?;
                cols.gv.extend(c.gv);
                cols.ngv.extend(c.ngv);
                cols.gate.extend(c.gate);
            }
            let g = gate_same(&all, &cols);
            println!(
                "mcol gate m={m} vs {m} m=1 launches: bit_identical={g} {}",
                verdict(g)
            );
            out.push(g);
        }
        // From 500 a batch of 8; from 1,000 a batch of 40 whose slots wrap the
        // ring twice (1000 mod 17 = 14) and whose first nine rows read the ring.
        for (m, p0) in [(8usize, 500u32), (40, 1_000)] {
            let ring = Lcg(0x6d72 + m as u64).fill(ring_len(HC), -2.0, 2.0);
            let ci = ConvIn::new(m, 0x6d6376 + m as u64);
            let one = cx.conv(unl, &ci, &positions(p0, m), &ring)?;
            let chained = cx.conv_chained(&ci, p0, &ring)?;
            let c = conv_same(&one, &chained);
            println!(
                "mcol conv m={m} from {p0} vs {m} one-token launches through the ring: \
                 bit_identical out+ring={c} {}",
                verdict(c)
            );
            out.push(c);
        }
        Ok(out)
    }

    // --------------------------------------------------------- 4. wrap

    /// Clause 4.
    fn wrap(cx: &Cx<'_>) -> Result<bool, GateError> {
        let unl = cx.gpu.unlabelled_sink();
        let m = 41;
        let ring = vec![0.0f32; ring_len(HC)];
        let i = ConvIn::new(m, 0x7772);
        let pos = positions(0, m);
        let one = cx.conv(unl, &i, &pos, &ring)?;
        let chained = cx.conv_chained(&i, 0, &ring)?;
        let host = i.host(&pos, &ring);
        let same_at = |t: usize| {
            bits_equal(
                &chained.out[t * CH..(t + 1) * CH],
                &host.out[t * CH..(t + 1) * CH],
            )
        };
        let marks: Vec<String> = [8usize, 9, 10, 16, 17, 18, 33, 34, 35]
            .iter()
            .map(|&t| format!("{t}:{}", same_at(t)))
            .collect();
        let pass = conv_same(&one, &host) && conv_same(&chained, &host);
        println!(
            "wrap 41 tokens from 0, one launch and 41 chained vs host: bit_identical out+ring \
             (chained, per position {}) {}",
            marks.join(" "),
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------------ 5. rollback

    /// Clause 5.
    fn rollback(cx: &Cx<'_>) -> Result<bool, GateError> {
        let unl = cx.gpu.unlabelled_sink();
        let zero = vec![0.0f32; ring_len(HC)];
        let old = ConvIn::new(28, 0x726f);
        let a = cx.conv(unl, &old.rows(0, 20), &positions(0, 20), &zero)?;
        let pass8 = cx.conv(unl, &old.rows(20, 28), &positions(20, 8), &a.ring)?;
        let mut new = ConvIn::new(4, 0x6e77);
        new.w.clone_from(&old.w);
        let after = cx.conv(unl, &new, &positions(23, 4), &pass8.ring)?;
        let joined = old.then(23, &new);
        let host = joined.host(&positions(0, 27), &zero);
        let pass = bits_equal(&after.out, &host.out[23 * CH..27 * CH]);
        println!(
            "rollback pass 20..27, then 23..26 on the ring it left vs the host rule over 0..26 \
             from zero: bit_identical={pass} {}",
            verdict(pass)
        );
        Ok(pass)
    }

    // --------------------------------------------------------- 6. reset

    /// Clause 6.
    fn reset(cx: &Cx<'_>) -> Result<bool, GateError> {
        let s = cx.gpu.stream();
        let random = Lcg(0x7273).fill(ring_len(HC), -2.0, 2.0);
        let mut rd = DeviceBuffer::from_host(s, &random)?;
        cx.k.reset_ring(s, &mut rd, HC)?;
        s.synchronize()?;
        let zeroed = rd.to_host_vec(s)?.iter().all(|&v| v.to_bits() == 0);
        let unl = cx.gpu.unlabelled_sink();
        let i = ConvIn::new(6, 0x7274);
        let pos = positions(0, 6);
        let over_random = cx.conv(unl, &i, &pos, &random)?;
        let over_zero = cx.conv(unl, &i, &pos, &vec![0.0; ring_len(HC)])?;
        let start = bits_equal(&over_random.out, &over_zero.out);
        let pass = zeroed && start;
        println!(
            "reset ring all zero={zeroed}, position 0 over a random ring = over zeros={start} {}",
            verdict(pass)
        );
        Ok(pass)
    }

    // --------------------------------------------------------- 7. fault

    /// The judgement of one planted input: the word before, the word the
    /// planted launch left and the word after a clean rerun; which values
    /// the plant reaches, and both runs' values.
    fn judge_plant(
        what: &str,
        want: Fault,
        words: [Option<Fault>; 3],
        hit: &[usize],
        clean: &[f32],
        bad: &[f32],
    ) -> bool {
        let [before, word, after] = words;
        let hit_nan = hit.iter().all(|&j| bad[j].is_nan());
        let mut mask = vec![false; clean.len()];
        for &j in hit {
            mask[j] = true;
        }
        let rest = clean
            .iter()
            .zip(bad)
            .zip(&mask)
            .all(|((c, b), &h)| h || c.to_bits() == b.to_bits());
        let pass = before.is_none() && word == Some(want) && after.is_none() && hit_nan && rest;
        println!(
            "fault {what}: word={word:?} (want {want:?}) clean before={} after={} hit NaN={hit_nan} \
             ({} values) rest bit-identical={rest} {}",
            before.is_none(),
            after.is_none(),
            hit.len(),
            verdict(pass)
        );
        pass
    }

    /// Clause 7.
    fn fault_clauses(cx: &Cx<'_>) -> Result<Vec<bool>, GateError> {
        let unl = cx.gpu.unlabelled_sink();
        let mut out = Vec::new();
        let m = 3;
        let clean_in = GateIn::new(m, 0x6661);
        let clean = cx.gate(unl, &clean_in)?;
        let flat = |g: &GateOut| [g.gv.clone(), g.ngv.clone(), g.gate.clone()].concat();
        // The gate's outputs a block reaches: its gv row, its ngv row, its gate.
        let block = |b: usize| -> Vec<usize> {
            let (nv, ng) = (m * CH, 2 * m * CH);
            (b * ROW..(b + 1) * ROW)
                .chain(nv + b * ROW..nv + (b + 1) * ROW)
                .chain([ng + b])
                .collect()
        };
        let ngv_block = |b: usize| -> Vec<usize> {
            let nv = m * CH;
            (nv + b * ROW..nv + (b + 1) * ROW).collect()
        };
        for (layer, what, plant) in [
            (1usize, "gate NaN key value (token 1, stream 2)", 0usize),
            (
                2,
                "gate key row of 1e30, squares overflow (token 0, stream 1)",
                1,
            ),
            (
                3,
                "gate infinite value (token 2): every stream's gv norm",
                2,
            ),
        ] {
            let sink = cx.gpu.layer_sink(layer)?;
            let before = cx.gpu.fault()?;
            let mut bad_in = clean_in.clone();
            let hit: Vec<usize> = match plant {
                0 => {
                    bad_in.key[(HC + 2) * ROW + 77] = f32::NAN;
                    block(HC + 2)
                }
                1 => {
                    bad_in.key[ROW..2 * ROW].fill(1e30);
                    block(1)
                }
                _ => {
                    bad_in.value[2 * ROW + 5] = f32::INFINITY;
                    (2 * HC..3 * HC).flat_map(ngv_block).collect()
                }
            };
            let bad = cx.gate(sink, &bad_in)?;
            let word = cx.gpu.take_fault()?;
            let _ = cx.gate(sink, &clean_in)?;
            let after = cx.gpu.fault()?;
            let want = Fault::at(u32::try_from(layer)?, FaultSite::Ple);
            let (bad_flat, clean_flat) = (flat(&bad), flat(&clean));
            // An infinite value's own gv column is inf times a finite gate, not
            // NaN: judged apart, and taken out of the bit-identical rest.
            let pass = if plant == 2 {
                let inf_col: Vec<usize> = (0..HC).map(|s| (2 * HC + s) * ROW + 5).collect();
                let infinite = inf_col.iter().all(|&j| bad_flat[j].is_infinite());
                let mut masked_bad = bad_flat.clone();
                for &j in &inf_col {
                    masked_bad[j] = clean_flat[j];
                }
                println!("fault {what}: gv at the infinite column infinite={infinite}");
                infinite
                    && judge_plant(
                        what,
                        want,
                        [before, word, after],
                        &hit,
                        &clean_flat,
                        &masked_bad,
                    )
            } else {
                judge_plant(
                    what,
                    want,
                    [before, word, after],
                    &hit,
                    &clean_flat,
                    &bad_flat,
                )
            };
            out.push(pass);
        }

        let cm = 12;
        let ring = Lcg(0x6672).fill(ring_len(HC), -2.0, 2.0);
        let ci = ConvIn::new(cm, 0x6663);
        let pos = positions(100, cm);
        let clean = cx.conv(unl, &ci, &pos, &ring)?;
        for (layer, what, plant) in [
            (4usize, "conv NaN input (token 1, channel 77)", 0usize),
            (5, "conv position not p0 + t (token 2)", 1),
        ] {
            let sink = cx.gpu.layer_sink(layer)?;
            let before = cx.gpu.fault()?;
            let mut bad_in = ci.clone();
            let mut bad_pos = pos.clone();
            let hit: Vec<usize> = match plant {
                0 => {
                    bad_in.ngv[CH + 77] = f32::NAN;
                    // Read at distances 0, 3, 6, 9 by tokens 1, 4, 7, 10.
                    [1usize, 4, 7, 10].iter().map(|t| t * CH + 77).collect()
                }
                _ => {
                    bad_pos[2] += 1;
                    (2 * CH..3 * CH).collect()
                }
            };
            let bad = cx.conv(sink, &bad_in, &bad_pos, &ring)?;
            let word = cx.gpu.take_fault()?;
            let _ = cx.conv(sink, &ci, &pos, &ring)?;
            let after = cx.gpu.fault()?;
            let want = Fault::at(u32::try_from(layer)?, FaultSite::Ple);
            out.push(judge_plant(
                what,
                want,
                [before, word, after],
                &hit,
                &clean.out,
                &bad.out,
            ));
        }
        Ok(out)
    }

    // -------------------------------------------------------- 8. refuse

    /// Clause 8.
    fn refuse(cx: &Cx<'_>) -> Result<bool, GateError> {
        let s = cx.gpu.stream();
        let unl = cx.gpu.unlabelled_sink();
        let z = |n: usize| DeviceBuffer::<f32>::zeroed(s, n);
        let (ngv, gv, x, w) = (z(CH)?, z(CH)?, z(CH)?, z(TAPS * CH)?);
        let pos = DeviceBuffer::from_host(s, &[0u32])?;
        let mut out = z(CH)?;
        let mut ring = z(ring_len(HC))?;
        let mut short_ring = z(ring_len(HC) - 1)?;
        let mut conv = |taps: usize, dilation: usize, hc: usize, m: usize, short: bool| {
            let ring = if short { &mut short_ring } else { &mut ring };
            cx.k.enqueue_conv(
                s,
                PleConvArgs {
                    ngv: &ngv,
                    gv: &gv,
                    x: &x,
                    w: &w,
                    pos: &pos,
                    taps,
                    dilation,
                    hc,
                    m,
                    fault: unl,
                    out: &mut out,
                    ring,
                },
            )
            .err()
            .map(|e| e.to_string())
        };
        let cases = [
            (
                "taps 3",
                conv(3, DILATION, HC, 1, false),
                "4 taps 3 positions apart",
            ),
            (
                "dilation 2",
                conv(TAPS, 2, HC, 1, false),
                "4 taps 3 positions apart",
            ),
            ("hc 0", conv(TAPS, DILATION, 0, 1, false), "need hc >= 1"),
            ("m 0", conv(TAPS, DILATION, HC, 0, false), "m >= 1"),
            (
                "short ring",
                conv(TAPS, DILATION, HC, 1, true),
                "ring.len()",
            ),
        ];
        let (key, value) = (z(CH - 1)?, z(ROW)?);
        let (g3, mut o1, mut o2, mut o3) = (z(CH)?, z(CH)?, z(CH)?, z(HC)?);
        let short_key =
            cx.k.enqueue_gate(
                s,
                PleGateArgs {
                    key: &key,
                    value: &value,
                    x: &x,
                    gain_key: &g3,
                    gain_query: &g3,
                    gain_conv: &g3,
                    eps: EPS,
                    hc: HC,
                    m: 1,
                    fault: unl,
                    gv: &mut o1,
                    ngv: &mut o2,
                    gate: &mut o3,
                },
            )
            .err()
            .map(|e| e.to_string());
        let mut pass = true;
        for (what, got, want) in cases
            .into_iter()
            .chain([("short key", short_key, "key.len()")])
        {
            let ok = got.as_deref().is_some_and(|e| e.contains(want));
            println!(
                "refuse {what}: {got:?} (want it to name {want:?}) {}",
                verdict(ok)
            );
            pass &= ok;
        }
        Ok(pass)
    }
}
