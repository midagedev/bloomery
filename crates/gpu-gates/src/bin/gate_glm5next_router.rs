//! GPU gate for the sigmoid routers of `bloomery_gpu_deepseek41::router`,
//! GLM-5.3-Flash's (`glm5next`) and MiMo-V2.6-Flash's (`mimo2`): the experts
//! scored `sigmoid(logit)`, a selection bias for the selection only, the top
//! 8 by (value descending, index descending), the unbiased scores summed in
//! f64 in slot order, narrowed, guarded (`renorm_divisor`), each divided by
//! it and scaled by the file's `expert_weights_scale`. Model-free: the router
//! weight, the activations and the bias are seeded, so the gate needs no file
//! and no reference set.
//!
//! The clauses below run once per instance, as its own cases, over the same
//! host rule: `glm5next` at 288 experts and a scale of 2.5, then `mimo2` at
//! 256 experts and a scale of 1. A line names its instance by the entry in
//! `ties case=` and `graph op=`; the instance's header line opens its
//! clauses. The entry names below are `glm5next`'s; `mimo2`'s are the
//! same with the prefix `mimo2_`.
//!
//! One clause per rule, each its own verdict line:
//! - `logits`: [`TOKENS`] one-token launches (`glm5next_router`) at
//!   K = [`K`]; every logit BIT-EQUAL to the host transcription of
//!   `f32_gemv`'s m = 1 row (the lane walk by `mul_add` from 0, the xor
//!   butterfly), the ticket count zero after every launch.
//! - `scores`: every score within the band of the host `route_core::sigmoid`
//!   of the kernel's own logit. The two sides share the formula `1 / (1 +
//!   e^-x)` and differ in `expf` only; the band at a logit is the sum of the
//!   two sides' distances from the exact value ([`sigmoid_err`]: the `expf`
//!   error, [`DEVICE_EXP_ULPS`] or [`HOST_EXP_ULPS`], and half an ulp for
//!   `1 + e` carried through the reciprocal, half an ulp for the quotient).
//! - `ids`: every token's eight ids EQUAL to the host's top 8 of the kernel's
//!   scores plus the bias, by (value descending, index descending).
//! - `weights`: every token's eight weights BIT-EQUAL to the host rule on the
//!   kernel's scores at the kernel's ids.
//! - `batch`: the two batch launches (`glm5next_router_scores`, then
//!   `glm5next_router_pick`) over all [`TOKENS`] tokens — eight token tiles,
//!   the last partial — give every token's scores, ids and weights BIT-EQUAL
//!   to its one-token launch, and a rerun bit-identical.
//! - `ties`: planted logits (a K = 32 weight whose first column is the logit,
//!   against a one-hot input, so every logit is exact) where the selection
//!   order decides: two experts of one lane tied in the top 8, the 8th place
//!   tied, adjacent lanes, the two ends, three ways, every logit equal, the
//!   bias deciding, and every score 0 (logits of −200, where `e^200`
//!   overflows): the ids EQUAL to the rule's statement on the exact keys
//!   (sigmoid in f64 plus the bias) and to the host selection on the
//!   kernel's scores, the weights BIT-EQUAL to the host rule and finite, no
//!   fault raised.
//! - `guard`: every logit −40, so the eight scores sum below 2^-42: the
//!   weights BIT-EQUAL to the guarded rule, finite; the line prints how many
//!   differ from the bare-sum rule (ik's), the named difference.
//! - `fault`: a NaN in router row 17, and then a −inf in bias entry 200, each
//!   through the one-token launch and through the batch pair: every launch
//!   raises `FaultSite::Router` (the unlabelled layer), and a clean launch
//!   raises nothing.
//! - `graph`: the one-token launch captured as a graph: one node, two
//!   replays bit-identical to the eager launch, the ticket count zero after
//!   each.
//!
//! What the gate does not pin: ik's ids (`ffn_moe_topk`) on the model's own
//! activations — that clause needs the `glm5next` reference set and lands
//! with the engine's program.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_glm5next_router: built without the `deepseek41` feature; see `just gate-gpu-glm-router`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_glm5next_router", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use bloomery_gpu::route_core::{renorm_divisor, sigmoid};
    use bloomery_gpu::{DeviceTensor, Fault, FaultSink, FaultSite, Gpu, GpuError, LAYER_NONE};
    use bloomery_gpu_deepseek41::router::{glm5next, mimo2};
    use bloomery_gpu_gates::rounding::butterfly;
    use bloomery_gpu_gates::{GateError, bits_equal, checks_failed, verdict};
    use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
    use std::sync::Arc;

    const NAME: &str = "gate_glm5next_router";
    /// The router's input width: the model's hidden size.
    const K: usize = 4096;
    /// Tokens of the seeded clauses: seven full tiles of the batch score
    /// launch and a partial eighth.
    const TOKENS: usize = 61;
    /// Experts each token routes to: both instances keep eight.
    const N_USED: usize = 8;
    const _: () = assert!(glm5next::N_USED == N_USED && mimo2::N_USED == N_USED);
    /// CUDA's `expf` (the libdevice body the kernel calls): at most 2 ulp
    /// (CUDA C++ Programming Guide, mathematical functions), and `1 + expf`
    /// may be fused into its last step and round once — 2.5 ulp of `e^-x`,
    /// the bound `gate_deepseek41_moe` states for the same call.
    const DEVICE_EXP_ULPS: f64 = 2.5;
    /// The host's `expf` (glibc): at most 1 ulp, glibc's documented bound.
    const HOST_EXP_ULPS: f64 = 1.0;

    /// A router instance under the gate: its module's types, entries and the
    /// file numbers its rule is stated at. The clauses are generic over it, so
    /// one clause list runs at each instance's own numbers.
    trait Inst {
        /// The one-token entry's name.
        const OP: &'static str;
        /// Experts the instance routes over.
        const E: usize;
        /// The instance's `expert_weights_scale`.
        const SCALE: f32;
        type Kernels;
        type Out;
        fn load(ctx: &Arc<CudaContext>) -> Result<Self::Kernels, GpuError>;
        fn new_out(s: &CudaStream) -> Result<Self::Out, GpuError>;
        #[allow(clippy::too_many_arguments, reason = "the module's launcher, flat")]
        fn enqueue_router(
            k: &Self::Kernels,
            s: &CudaStream,
            w: &DeviceTensor<f32>,
            x: &DeviceBuffer<f32>,
            bias: &DeviceBuffer<f32>,
            out: &mut Self::Out,
            fault: FaultSink,
        ) -> Result<(), GpuError>;
        #[allow(clippy::too_many_arguments, reason = "the module's launcher, flat")]
        fn enqueue_rows(
            k: &Self::Kernels,
            s: &CudaStream,
            w: &DeviceTensor<f32>,
            x: &DeviceBuffer<f32>,
            bias: &DeviceBuffer<f32>,
            t: usize,
            probs: &mut DeviceBuffer<f32>,
            ids: &mut DeviceBuffer<u32>,
            weights: &mut DeviceBuffer<f32>,
            fault: FaultSink,
        ) -> Result<(), GpuError>;
        /// The launch's logits, scores, ids and weights.
        #[allow(clippy::type_complexity, reason = "one gate helper's four readbacks")]
        fn read(
            out: &Self::Out,
            s: &CudaStream,
        ) -> Result<(Vec<f32>, Vec<f32>, Vec<u32>, Vec<f32>), GateError>;
        /// Zero the four result buffers.
        fn zero(out: &mut Self::Out, s: &CudaStream) -> Result<(), GateError>;
        /// The block ticket count as it stands.
        fn tickets(out: &Self::Out, s: &CudaStream) -> Result<u32, GateError>;
    }

    /// `$ty` as the instance of module `$m`, entry `$op`, at `$scale`.
    macro_rules! instance {
        ($ty:ident, $m:ident, $op:literal, $scale:expr) => {
            struct $ty;

            impl Inst for $ty {
                const OP: &'static str = $op;
                const E: usize = $m::N_EXPERT;
                const SCALE: f32 = $scale;
                type Kernels = $m::RouterKernels;
                type Out = $m::RouterOut;

                fn load(ctx: &Arc<CudaContext>) -> Result<Self::Kernels, GpuError> {
                    $m::RouterKernels::load(ctx)
                }

                fn new_out(s: &CudaStream) -> Result<Self::Out, GpuError> {
                    $m::RouterOut::new(s)
                }

                fn enqueue_router(
                    k: &Self::Kernels,
                    s: &CudaStream,
                    w: &DeviceTensor<f32>,
                    x: &DeviceBuffer<f32>,
                    bias: &DeviceBuffer<f32>,
                    out: &mut Self::Out,
                    fault: FaultSink,
                ) -> Result<(), GpuError> {
                    k.enqueue_router(s, w, x, bias, Self::SCALE, out, fault)
                }

                fn enqueue_rows(
                    k: &Self::Kernels,
                    s: &CudaStream,
                    w: &DeviceTensor<f32>,
                    x: &DeviceBuffer<f32>,
                    bias: &DeviceBuffer<f32>,
                    t: usize,
                    probs: &mut DeviceBuffer<f32>,
                    ids: &mut DeviceBuffer<u32>,
                    weights: &mut DeviceBuffer<f32>,
                    fault: FaultSink,
                ) -> Result<(), GpuError> {
                    k.enqueue_router_rows(s, w, x, bias, Self::SCALE, t, probs, ids, weights, fault)
                }

                fn read(
                    out: &Self::Out,
                    s: &CudaStream,
                ) -> Result<(Vec<f32>, Vec<f32>, Vec<u32>, Vec<f32>), GateError> {
                    Ok((
                        out.logits.to_host_vec(s)?,
                        out.probs.to_host_vec(s)?,
                        out.ids.to_host_vec(s)?,
                        out.weights.to_host_vec(s)?,
                    ))
                }

                fn zero(out: &mut Self::Out, s: &CudaStream) -> Result<(), GateError> {
                    out.logits.zero_async(s)?;
                    out.probs.zero_async(s)?;
                    out.ids.zero_async(s)?;
                    out.weights.zero_async(s)?;
                    Ok(())
                }

                fn tickets(out: &Self::Out, s: &CudaStream) -> Result<u32, GateError> {
                    Ok(out.tickets(s)?)
                }
            }
        };
    }

    // GLM-5.3-Flash: 288 experts, the file's scale 2.5.
    instance!(Glm, glm5next, "glm5next_router", 2.5);
    // MiMo-V2.6-Flash: 256 experts, the file's scale 1.
    instance!(Mimo, mimo2, "mimo2_router", 1.0);

    /// One f32 ulp at `v`: the spacing of f32 values at `|v|`'s binade (the
    /// subnormal spacing below the normal range).
    fn ulp32(v: f64) -> f64 {
        let a = v.abs() as f32;
        if a < f32::MIN_POSITIVE {
            return f64::from(f32::from_bits(1));
        }
        f64::from(f32::from_bits(a.to_bits() & 0x7f80_0000)) * f64::from(f32::EPSILON)
    }

    /// One side's distance from the exact `1 / (1 + e^-x)`: `e = e^-x` within
    /// `e_ulps` ulp, `y = 1 + e` rounded once more (half an ulp), `1 / y`
    /// moved by at most `dy / (y (y − dy))`, and the quotient rounded (half
    /// an ulp).
    fn sigmoid_err(x: f32, e_ulps: f64) -> f64 {
        let e = (-f64::from(x)).exp();
        let y = 1.0 + e;
        let dy = e_ulps * ulp32(e) + 0.5 * ulp32(y);
        dy / (y * (y - dy)) + 0.5 * ulp32(1.0 / y)
    }

    /// `n` values uniform on `[lo, lo + span)` from a 32-bit LCG.
    fn seeded(n: usize, seed: u32, lo: f32, span: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                lo + ((s >> 8) as f32 / (1u32 << 24) as f32) * span
            })
            .collect()
    }

    /// One lane's `f32_gemv` partial: `w[32·it + lane] · x[32·it + lane]` by
    /// `mul_add` from 0, `it` ascending.
    fn gemv_lane(w: &[f32], x: &[f32], lane: usize) -> f32 {
        let mut f = 0.0f32;
        let mut i = lane;
        while i < x.len() {
            f = w[i].mul_add(x[i], f);
            i += 32;
        }
        f
    }

    /// The top [`N_USED`] of `key`, descending, an equal key to the larger id.
    fn top_by<T: Copy>(key: &[T], cmp: impl Fn(&T, &T) -> std::cmp::Ordering) -> Vec<u32> {
        let mut idx: Vec<u32> = (0..key.len() as u32).collect();
        idx.sort_by(|&a, &b| cmp(&key[b as usize], &key[a as usize]).then(b.cmp(&a)));
        idx.truncate(N_USED);
        idx
    }

    /// The host selection on scores `p` and `bias`: the top [`N_USED`] of
    /// `p + bias` in f32.
    fn select(p: &[f32], bias: &[f32]) -> Vec<u32> {
        let v: Vec<f32> = p.iter().zip(bias).map(|(&p, &b)| p + b).collect();
        top_by(&v, f32::total_cmp)
    }

    /// The weights rule on scores `p` at `ids`: summed in f64 in slot order,
    /// narrowed, guarded ([`renorm_divisor`]), each divided by it, scaled.
    fn weights_rule(p: &[f32], ids: &[u32], scale: f32) -> Vec<f32> {
        let g: Vec<f32> = ids.iter().map(|&i| p[i as usize]).collect();
        let div = renorm_divisor(g.iter().fold(0.0f64, |s, &v| s + f64::from(v)) as f32);
        g.iter().map(|&v| v / div * scale).collect()
    }

    /// The same without the guard: ik's bare sum.
    fn weights_bare(p: &[f32], ids: &[u32], scale: f32) -> Vec<f32> {
        let g: Vec<f32> = ids.iter().map(|&i| p[i as usize]).collect();
        let div = g.iter().fold(0.0f64, |s, &v| s + f64::from(v)) as f32;
        g.iter().map(|&v| v / div * scale).collect()
    }

    /// `1 / (1 + e^-x)` in f64: the exact key of a planted logit.
    fn sigmoid64(x: f32) -> f64 {
        1.0 / (1.0 + (-f64::from(x)).exp())
    }

    /// Distance in f32 ulps (finite values of one sign).
    fn ulps(a: f32, b: f32) -> u32 {
        a.to_bits().abs_diff(b.to_bits())
    }

    struct Cx<I: Inst> {
        gpu: Gpu,
        rk: I::Kernels,
    }

    /// What one-token launches leave, token-major: logits and scores
    /// the instance's expert count a token, ids and weights [`N_USED`] a token, and the
    /// ticket count after each launch.
    struct Routed {
        logits: Vec<f32>,
        probs: Vec<f32>,
        ids: Vec<u32>,
        weights: Vec<f32>,
        tickets: Vec<u32>,
    }

    /// One `glm5next_router` launch per token of `x` (`k` a token).
    fn one_token_runs<I: Inst>(
        cx: &Cx<I>,
        w: &DeviceTensor<f32>,
        x: &[f32],
        k: usize,
        bias: &DeviceBuffer<f32>,
    ) -> Result<Routed, GateError> {
        let s = cx.gpu.stream();
        let mut out = I::new_out(s)?;
        let mut r = Routed {
            logits: Vec::new(),
            probs: Vec::new(),
            ids: Vec::new(),
            weights: Vec::new(),
            tickets: Vec::new(),
        };
        for xt in x.chunks_exact(k) {
            let xd = DeviceBuffer::from_host(s, xt)?;
            I::enqueue_router(&cx.rk, s, w, &xd, bias, &mut out, cx.gpu.unlabelled_sink())?;
            s.synchronize()?;
            let (logits, probs, ids, weights) = I::read(&out, s)?;
            r.logits.extend(logits);
            r.probs.extend(probs);
            r.ids.extend(ids);
            r.weights.extend(weights);
            r.tickets.push(I::tickets(&out, s)?);
        }
        Ok(r)
    }

    /// The batch pair over every token of `x`: scores, ids, weights.
    #[allow(clippy::type_complexity, reason = "one gate helper's three readbacks")]
    fn batch_run<I: Inst>(
        cx: &Cx<I>,
        w: &DeviceTensor<f32>,
        x: &[f32],
        k: usize,
        bias: &DeviceBuffer<f32>,
    ) -> Result<(Vec<f32>, Vec<u32>, Vec<f32>), GateError> {
        let s = cx.gpu.stream();
        let t = x.len() / k;
        let xd = DeviceBuffer::from_host(s, x)?;
        let mut probs = DeviceBuffer::<f32>::zeroed(s, I::E * t)?;
        let mut ids = DeviceBuffer::<u32>::zeroed(s, N_USED * t)?;
        let mut weights = DeviceBuffer::<f32>::zeroed(s, N_USED * t)?;
        I::enqueue_rows(
            &cx.rk,
            s,
            w,
            &xd,
            bias,
            t,
            &mut probs,
            &mut ids,
            &mut weights,
            cx.gpu.unlabelled_sink(),
        )?;
        s.synchronize()?;
        Ok((
            probs.to_host_vec(s)?,
            ids.to_host_vec(s)?,
            weights.to_host_vec(s)?,
        ))
    }

    /// The seeded router: weight, activations, bias.
    struct Seeded {
        w: Vec<f32>,
        x: Vec<f32>,
        bias: Vec<f32>,
    }

    fn seeded_router<I: Inst>() -> Seeded {
        Seeded {
            w: seeded(I::E * K, 7, -0.05, 0.1),
            x: seeded(TOKENS * K, 11, -2.0, 4.0),
            bias: seeded(I::E, 13, -0.25, 0.5),
        }
    }

    /// Clauses `logits`, `scores`, `ids`, `weights`, `batch`, with `r` the
    /// one-token launches over the seeded router.
    fn seeded_clauses<I: Inst>(
        cx: &Cx<I>,
        sd: &Seeded,
        w: &DeviceTensor<f32>,
        bias: &DeviceBuffer<f32>,
        r: &Routed,
    ) -> Result<bool, GateError> {
        let (mut logits_ok, mut ids_ok, mut weights_ok) = (0usize, 0usize, 0usize);
        let (mut worst, mut max_ulps) = (0.0f64, 0u32);
        for t in 0..TOKENS {
            let xt = &sd.x[t * K..(t + 1) * K];
            let e_rows = t * I::E..(t + 1) * I::E;
            let s_rows = t * N_USED..(t + 1) * N_USED;
            let lh: Vec<f32> = (0..I::E)
                .map(|e| {
                    butterfly(std::array::from_fn(|lane| {
                        gemv_lane(&sd.w[e * K..(e + 1) * K], xt, lane)
                    }))
                })
                .collect();
            let (lk, pk) = (&r.logits[e_rows.clone()], &r.probs[e_rows]);
            let (ik, wk) = (&r.ids[s_rows.clone()], &r.weights[s_rows]);
            logits_ok += usize::from(bits_equal(lk, &lh));
            for (&p, &l) in pk.iter().zip(lk) {
                let h = sigmoid(l);
                let band = sigmoid_err(l, DEVICE_EXP_ULPS) + sigmoid_err(l, HOST_EXP_ULPS);
                worst = worst.max((f64::from(p) - f64::from(h)).abs() / band);
                max_ulps = max_ulps.max(ulps(p, h));
            }
            ids_ok += usize::from(select(pk, &sd.bias) == ik);
            weights_ok += usize::from(bits_equal(wk, &weights_rule(pk, ik, I::SCALE)));
        }
        let tickets_ok = r.tickets.iter().all(|&c| c == 0);
        let mut ok = true;
        let pass = logits_ok == TOKENS && tickets_ok;
        println!(
            "logits tokens={TOKENS} K={K} bit_identical_f32_gemv={logits_ok}/{TOKENS} tickets_zero={tickets_ok} {}",
            verdict(pass)
        );
        ok &= pass;
        let pass = worst <= 1.0;
        println!(
            "scores tokens={TOKENS} vs host sigmoid of the kernel's logits: max |d|/band {worst:.3} \
             (expf {DEVICE_EXP_ULPS} + {HOST_EXP_ULPS} ulp), max {max_ulps} ulp {}",
            verdict(pass)
        );
        ok &= pass;
        let pass = ids_ok == TOKENS;
        println!(
            "ids tokens={TOKENS} equal to the host top {N_USED} of score + bias (ties to the larger id): \
             {ids_ok}/{TOKENS} {}",
            verdict(pass)
        );
        ok &= pass;
        let pass = weights_ok == TOKENS;
        let scale = I::SCALE;
        println!(
            "weights tokens={TOKENS} bit-identical to the host rule (f64 slot sum, guard, divide, x{scale}): \
             {weights_ok}/{TOKENS} {}",
            verdict(pass)
        );
        ok &= pass;

        let (pb, ib, wb) = batch_run(cx, w, &sd.x, K, bias)?;
        let (pb2, ib2, wb2) = batch_run(cx, w, &sd.x, K, bias)?;
        let same = (0..TOKENS)
            .filter(|&t| {
                let (e, s) = (t * I::E..(t + 1) * I::E, t * N_USED..(t + 1) * N_USED);
                bits_equal(&pb[e.clone()], &r.probs[e])
                    && ib[s.clone()] == r.ids[s.clone()]
                    && bits_equal(&wb[s.clone()], &r.weights[s])
            })
            .count();
        let rerun = bits_equal(&pb, &pb2) && ib == ib2 && bits_equal(&wb, &wb2);
        let pass = same == TOKENS && rerun;
        println!(
            "batch tokens={TOKENS} (scores + pick) vs the one-token launch: {same}/{TOKENS} tokens \
             bit-identical, rerun_bit_identical={rerun} {}",
            verdict(pass)
        );
        ok &= pass;
        Ok(ok)
    }

    /// The router of planted `logits` (the first column of a K = 32 weight
    /// against a one-hot input) and `bias`, one token: what it leaves.
    fn planted<I: Inst>(cx: &Cx<I>, logits: &[f32], bias: &[f32]) -> Result<Routed, GateError> {
        const PK: usize = 32;
        let s = cx.gpu.stream();
        let mut w = vec![0.0f32; I::E * PK];
        for (e, &v) in logits.iter().enumerate() {
            w[e * PK] = v;
        }
        let mut x = vec![0.0f32; PK];
        x[0] = 1.0;
        let wd = DeviceTensor::upload(s, &w, I::E, PK)?;
        let bd = DeviceBuffer::from_host(s, bias)?;
        one_token_runs(cx, &wd, &x, PK, &bd)
    }

    /// Clause `ties`: planted ties against the rule's statement.
    fn ties<I: Inst>(cx: &Cx<I>) -> Result<bool, GateError> {
        let base: Vec<f32> = (0..I::E).map(|e| -3.0 + 0.01 * e as f32).collect();
        let plant = |pairs: &[(usize, f32)]| {
            let mut v = base.clone();
            for &(i, val) in pairs {
                v[i] = val;
            }
            v
        };
        let zero = vec![0.0f32; I::E];
        let mut bias_decides = zero.clone();
        bias_decides[7] = 0.5;
        bias_decides[77] = 0.5;
        let cases: Vec<(&str, Vec<f32>, Vec<f32>)> = vec![
            (
                "one_lane_in_top8",
                plant(&[(10, 10.0), (42, 10.0)]),
                zero.clone(),
            ),
            (
                "boundary_8_9",
                plant(&[
                    (200, 6.0),
                    (201, 6.1),
                    (202, 6.2),
                    (203, 6.3),
                    (204, 6.4),
                    (205, 6.5),
                    (206, 6.6),
                    (100, 5.0),
                    (250, 5.0),
                ]),
                zero.clone(),
            ),
            (
                "adjacent_lanes",
                plant(&[(64, 7.0), (65, 7.0)]),
                zero.clone(),
            ),
            ("ends", plant(&[(0, 7.0), (I::E - 1, 7.0)]), zero.clone()),
            (
                "three_way",
                plant(&[(5, 7.0), (37, 7.0), (200, 7.0)]),
                zero.clone(),
            ),
            ("all_equal", vec![1.0; I::E], zero.clone()),
            ("bias_decides", vec![1.0; I::E], bias_decides),
            ("all_zero_scores", vec![-200.0; I::E], zero.clone()),
        ];
        let mut all = true;
        for (name, logits, bias) in cases {
            let r = planted(cx, &logits, &bias)?;
            let key: Vec<f64> = logits
                .iter()
                .zip(&bias)
                .map(|(&l, &b)| sigmoid64(l) + f64::from(b))
                .collect();
            let stated = top_by(&key, f64::total_cmp);
            let host = select(&r.probs, &bias);
            let w_ok = bits_equal(&r.weights, &weights_rule(&r.probs, &r.ids, I::SCALE));
            let finite = r.weights.iter().all(|w| w.is_finite());
            let fault = cx.gpu.take_fault()?;
            let pass = r.ids == stated
                && r.ids == host
                && w_ok
                && finite
                && bits_equal(&r.logits, &logits)
                && fault.is_none()
                && r.tickets == [0];
            println!(
                "ties case={name} ids={:?} stated={stated:?} host={host:?} weights_exact={w_ok} \
                 weights_finite={finite} fault={} tickets={:?} {}",
                r.ids,
                fault.map_or("none".to_string(), |f| f.to_string()),
                r.tickets,
                verdict(pass)
            );
            all &= pass;
        }
        Ok(all)
    }

    /// Clause `guard`: every logit −40, eight scores summing below 2^-42.
    fn guard<I: Inst>(cx: &Cx<I>) -> Result<bool, GateError> {
        let r = planted(cx, &vec![-40.0; I::E], &vec![0.0; I::E])?;
        let sum: f64 = r.ids.iter().map(|&i| f64::from(r.probs[i as usize])).sum();
        let w_ok = bits_equal(&r.weights, &weights_rule(&r.probs, &r.ids, I::SCALE));
        let finite = r.weights.iter().all(|w| w.is_finite());
        let bare = weights_bare(&r.probs, &r.ids, I::SCALE);
        let differ = r
            .weights
            .iter()
            .zip(&bare)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let fault = cx.gpu.take_fault()?;
        let pass = w_ok && finite && sum < 2f64.powi(-42) && fault.is_none() && r.tickets == [0];
        println!(
            "guard logits=-40 selected_sum={sum:.3e} (< 2^-42) weights_bit_identical_guarded_rule={w_ok} \
             finite={finite} differ_from_bare_sum={differ}/{N_USED} (the named difference, printed) \
             fault={} {}",
            fault.map_or("none".to_string(), |f| f.to_string()),
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `fault`: a NaN router row and a −inf bias entry, each through
    /// both shapes, raise `Router`; a clean launch raises nothing.
    fn fault<I: Inst>(cx: &Cx<I>, sd: &Seeded) -> Result<bool, GateError> {
        let s = cx.gpu.stream();
        let x = &sd.x[..8 * K];
        let want = Some(Fault::at(LAYER_NONE, FaultSite::Router));
        let clean_w = DeviceTensor::upload(s, &sd.w, I::E, K)?;
        let clean_b = DeviceBuffer::from_host(s, &sd.bias)?;
        cx.gpu.take_fault()?;
        one_token_runs(cx, &clean_w, &x[..K], K, &clean_b)?;
        batch_run(cx, &clean_w, x, K, &clean_b)?;
        let clean = cx.gpu.take_fault()?;
        let mut nan_w = sd.w.clone();
        nan_w[17 * K + 5] = f32::NAN;
        let nan_w = DeviceTensor::upload(s, &nan_w, I::E, K)?;
        let mut inf_b = sd.bias.clone();
        inf_b[200] = f32::NEG_INFINITY;
        let inf_b = DeviceBuffer::from_host(s, &inf_b)?;
        let mut got = Vec::new();
        for (w, b) in [(&nan_w, &clean_b), (&clean_w, &inf_b)] {
            let r = one_token_runs(cx, w, &x[..K], K, b)?;
            got.push((cx.gpu.take_fault()?, r.tickets[0]));
            batch_run(cx, w, x, K, b)?;
            got.push((cx.gpu.take_fault()?, 0));
        }
        let pass = clean.is_none() && got.iter().all(|&(f, c)| f == want && c == 0);
        let show = |f: Option<Fault>| f.map_or("none".to_string(), |f| f.to_string());
        println!(
            "fault clean={} nan_row17: one-token={} batch={} | bias200=-inf: one-token={} batch={} \
             (want router) tickets={:?} {}",
            show(clean),
            show(got[0].0),
            show(got[1].0),
            show(got[2].0),
            show(got[3].0),
            [got[0].1, got[2].1],
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `graph`: the one-token launch as a captured graph.
    fn graph<I: Inst>(
        cx: &Cx<I>,
        sd: &Seeded,
        w: &DeviceTensor<f32>,
        bias: &DeviceBuffer<f32>,
    ) -> Result<bool, GateError> {
        let s = cx.gpu.stream();
        let xd = DeviceBuffer::from_host(s, &sd.x[..K])?;
        let mut out = I::new_out(s)?;
        I::enqueue_router(&cx.rk, s, w, &xd, bias, &mut out, cx.gpu.unlabelled_sink())?;
        s.synchronize()?;
        let eager = I::read(&out, s)?;
        I::zero(&mut out, s)?;
        s.synchronize()?;
        let sink = cx.gpu.unlabelled_sink();
        let g = cx
            .gpu
            .capture(|st| I::enqueue_router(&cx.rk, st, w, &xd, bias, &mut out, sink))?;
        let (mut replays, mut tickets) = (true, Vec::new());
        for _ in 0..2 {
            g.launch(s)?;
            s.synchronize()?;
            let now = I::read(&out, s)?;
            replays &= bits_equal(&now.0, &eager.0)
                && bits_equal(&now.1, &eager.1)
                && now.2 == eager.2
                && bits_equal(&now.3, &eager.3);
            tickets.push(I::tickets(&out, s)?);
        }
        let nodes = g.node_count();
        let pass = replays && nodes == 1 && tickets.iter().all(|&c| c == 0);
        println!(
            "graph op={} two_replays_bit_identical={replays} tickets_after={tickets:?} \
             graph_nodes={nodes} {}",
            I::OP,
            verdict(pass)
        );
        Ok(pass)
    }

    /// Every clause at instance `I`; whether all passed.
    fn run_instance<I: Inst>(gpu: Gpu) -> Result<(Gpu, bool), GateError> {
        println!(
            "== {} ({} experts, top {N_USED}, sigmoid, x{}) ==",
            I::OP,
            I::E,
            I::SCALE
        );
        let rk = I::load(gpu.context())?;
        let cx = Cx::<I> { gpu, rk };
        let s = cx.gpu.stream();
        let sd = seeded_router::<I>();
        let w = DeviceTensor::upload(s, &sd.w, I::E, K)?;
        let bias = DeviceBuffer::from_host(s, &sd.bias)?;
        let r = one_token_runs(&cx, &w, &sd.x, K, &bias)?;
        let seeded_fault = cx.gpu.take_fault()?;
        let mut ok = seeded_fault.is_none();
        if let Some(f) = seeded_fault {
            println!("seeded launches raised {f}, want none FAIL");
        }
        ok &= seeded_clauses(&cx, &sd, &w, &bias, &r)?;
        ok &= ties(&cx)?;
        ok &= guard(&cx)?;
        ok &= fault(&cx, &sd)?;
        ok &= graph(&cx, &sd, &w, &bias)?;
        Ok((cx.gpu, ok))
    }

    pub fn run() -> Result<(), GateError> {
        let (gpu, glm_ok) = run_instance::<Glm>(Gpu::new()?)?;
        let (_gpu, mimo_ok) = run_instance::<Mimo>(gpu)?;
        if glm_ok && mimo_ok {
            println!(
                "PASSED: {NAME} — {} ({}/{N_USED}, sigmoid, x{}) and {} ({}/{N_USED}, sigmoid, x{}), each with its batch \
                 pair: logits, ids and weights bit-identical to the host rule over {TOKENS} seeded tokens, \
                 scores in the sigmoid band, 8 tie cases, the guard, the fault, the graph",
                Glm::OP,
                Glm::E,
                Glm::SCALE,
                Mimo::OP,
                Mimo::E,
                Mimo::SCALE
            );
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
