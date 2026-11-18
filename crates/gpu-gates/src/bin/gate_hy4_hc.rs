//! GPU gate for Hy4-flash's common card kernels: the identity hyper-connections
//! (`hc_ident`: pre, head, post), the engram delta (`engram_delta`: rows norm,
//! kv bias, delta and mask, with the ROW 3072 key and gate kernels of
//! `engram_gate::row_3072`) and the 512/16 floor router
//! (`route_body::sigmoid_512_16`). Synthetic inputs, no model file: the rules
//! are llama.cpp's `build_hc_*`, `build_engram` and `build_moe_ffn` graphs
//! (PR #32517 at 0a054d0b), restated here in f64 and in f32 op by op.
//!
//! Three kinds of clause:
//! - **bound**: the kernel against the rule in f64, within a bound this file
//!   derives from the kernel's own rounding count (`u = 2^-24`; each bound's
//!   comment counts the roundings it comes from) — never a tuned band;
//! - **bit**: the kernel against the same rule in f32, every op rounded as the
//!   kernel's contract names it, bit for bit (the ops that have a single
//!   rounding: a bias add, the HC fold and post);
//! - **fence**: a non-finite value is refused by the fault word, a host
//!   launcher refuses a short buffer by name, a captured graph replays the
//!   eager run bit for bit.
//!
//! This gate needs an NVIDIA device.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_hy4_hc", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::engram_delta::{
        DeltaArgs, EngramDeltaKernels, GateOnlyArgs, KeyNormArgs, RowsArgs,
    };
    use bloomery_gpu::gemm::Gemm32Kernels;
    use bloomery_gpu::hc_ident::{HcIdentKernels, HeadArgs, PostArgs, PreArgs};
    use bloomery_gpu::route_body::sigmoid_512_16::{RouterKernels, RouterOut};
    use bloomery_gpu::routed_block::{RouteIn, RoutedRows, RouterEntry};
    use bloomery_gpu::{DeviceTensor, Fault, FaultSite, Gpu, GpuError, LAYER_NONE};
    use bloomery_gpu_gates::{GateError, bits_equal, checks_failed, verdict};
    use cuda_core::DeviceBuffer;

    /// One unit roundoff of f32.
    const U: f64 = 1.0 / 16_777_216.0;
    /// Second-order terms of every first-order bound below are at most
    /// `(rounding count · u)^2 ≈ 1e-9` of it; a bound carries this factor.
    const SECOND_ORDER: f64 = 1.0 + 1e-3;

    const N: usize = 3072;
    const HC: usize = 4;
    const MIX: usize = 2 * HC;
    const K: usize = HC * N;
    const COLS: usize = 16;
    const WIDTH: usize = 96;
    const EXPERTS: usize = 512;
    const USED: usize = 16;
    const ROUTER_K: usize = 64;
    const SCALE: f32 = 3.998;
    const FLOOR: f32 = 1.0 / 16384.0;

    struct Cx {
        gpu: Gpu,
        hc: HcIdentKernels,
        eng: EngramDeltaKernels,
        router: RouterKernels,
        wide: Gemm32Kernels,
    }

    /// A deterministic generator of values in `[-amp, amp)`.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 40) as f64 / 16_777_216.0) * 2.0 - 1.0
        }

        fn fill(&mut self, n: usize, amp: f64) -> Vec<f32> {
            (0..n).map(|_| (self.next() * amp) as f32).collect()
        }
    }

    fn clause(name: &str, pass: bool, detail: &str) -> bool {
        println!("{name} {detail} {}", verdict(pass));
        pass
    }

    fn sigmoid(v: f64) -> f64 {
        1.0 / (1.0 + (-v).exp())
    }

    /// Every `|dev − want| <= bound` (an exact `want` has bound 0), and the
    /// largest `|dev − want| / bound` over the values with a bound.
    fn within(dev: &[f32], want: &[f64], bound: &[f64]) -> (bool, f64) {
        let mut worst = 0.0f64;
        let mut ok = dev.len() == want.len() && want.len() == bound.len();
        if !ok {
            return (false, f64::INFINITY);
        }
        for ((&d, &w), &b) in dev.iter().zip(want).zip(bound) {
            let err = (f64::from(d) - w).abs();
            if !d.is_finite() || err > b {
                ok = false;
            }
            if b > 0.0 {
                worst = worst.max(err / b);
            }
        }
        (ok, worst)
    }

    fn some_fault(cx: &Cx) -> Result<Option<Fault>, GateError> {
        Ok(cx.gpu.take_fault()?)
    }

    fn norm_fault() -> Option<Fault> {
        Some(Fault::at(LAYER_NONE, FaultSite::F32Product))
    }

    // ---------------------------------------------------------------- HC

    /// The rule of one token's mix in f64: the dots of `w`'s rows with the
    /// token's flat streams `x`, the norm scale, and per row the value
    /// `v_r = scale·dot_r·s_r + base_r` with its error bound.
    ///
    /// The bound counts the roundings of the kernel (`hc_ident`'s contract):
    /// - each dot: a thread's `K/256` fused multiply-adds, five butterfly
    ///   levels, three tree levels — `R = K/256 + 8` roundings, so the dot
    ///   is within `R·u·Σ|w·x|`;
    /// - the squares sum the same way, all terms positive: `R·u` relative;
    ///   the mean divides (u), eps adds (u): `(R + 2)·u` on the mean; the
    ///   scale `1/√mean` halves that and rounds twice (sqrt, divide):
    ///   `((R + 2)/2 + 2)·u`;
    /// - `scale·dot`, `·s_r`, `+ base_r`: three roundings, the add on `v_r`'s
    ///   own magnitude.
    struct MixRef {
        v: Vec<f64>,
        e_v: Vec<f64>,
    }

    fn mix_ref(x: &[f32], w: &[f32], rows: usize, s: &[f64], base: &[f32], eps: f64) -> MixRef {
        let r = (K / 256 + 8) as f64;
        let squares: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let scale = 1.0 / (squares / K as f64 + eps).sqrt();
        let scale_rel = ((r + 2.0) / 2.0 + 2.0) * U;
        let mut out = MixRef {
            v: vec![],
            e_v: vec![],
        };
        for row in 0..rows {
            let wr = &w[row * K..(row + 1) * K];
            let dot: f64 = wr
                .iter()
                .zip(x)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let abs: f64 = wr
                .iter()
                .zip(x)
                .map(|(&a, &b)| (f64::from(a) * f64::from(b)).abs())
                .sum();
            let sr = s[usize::from(row >= HC)];
            let t = scale * dot * sr;
            let v = t + f64::from(base[row]);
            let e =
                scale * sr.abs() * (r * U * abs) + t.abs() * (scale_rel + 2.0 * U) + U * v.abs();
            out.v.push(v);
            out.e_v.push(e);
        }
        out
    }

    /// The bound on `pre = σ(v) + eps` (`post = σ(v)·mag + eps`): `σ` is
    /// 1/4-Lipschitz, rounds three times (the f64 exponential's one rounding
    /// to f32, the `1 + e` add, the divide), and the last add is one more.
    fn gate_ref(v: f64, e_v: f64, mag: f64, eps: f64) -> (f64, f64) {
        let g = sigmoid(v);
        let e_g = g * (1.0 - g) * e_v + 3.0 * U;
        let val = g * mag + eps;
        (val, mag * e_g + 2.0 * U * val)
    }

    fn hc_clauses(cx: &Cx) -> Result<bool, GateError> {
        const M: usize = 3;
        let st = cx.gpu.stream();
        let mut rng = Rng(0x51ed_270b_1f0c_c0de);
        let mut x = rng.fill(M * K, 1.0);
        // Token 1's streams are 8x larger: the norm scale differs by token.
        for v in &mut x[K..2 * K] {
            *v *= 8.0;
        }
        let w = rng.fill(MIX * K, 0.05);
        let wh = rng.fill(HC * K, 0.05);
        let (scale_h, base_h) = ([1.5f32, 0.7], rng.fill(MIX, 1.0));
        let (scale_head, base_head) = ([1.25f32], rng.fill(HC, 1.0));
        // Distinct values: a swap of the two epsilons or a dropped one moves the
        // result by far more than the rounding bounds below.
        let (eps_norm, eps_hc, mag) = (1e-6f32, 0.05f32, 2.0f32);
        let dx = DeviceBuffer::from_host(st, &x)?;
        let dw = DeviceBuffer::from_host(st, &w)?;
        let dwh = DeviceBuffer::from_host(st, &wh)?;
        let (dscale, dbase) = (
            DeviceBuffer::from_host(st, &scale_h)?,
            DeviceBuffer::from_host(st, &base_h)?,
        );
        let (dscale_head, dbase_head) = (
            DeviceBuffer::from_host(st, &scale_head)?,
            DeviceBuffer::from_host(st, &base_head)?,
        );
        let mut mix = DeviceBuffer::zeroed(st, M * MIX)?;
        let mut y = DeviceBuffer::zeroed(st, M * N)?;
        let mut yh = DeviceBuffer::zeroed(st, M * N)?;
        cx.gpu.take_fault()?;
        let pre =
            |mix: &mut DeviceBuffer<f32>, y: &mut DeviceBuffer<f32>, src: &DeviceBuffer<f32>, m| {
                cx.hc.enqueue_pre(
                    st,
                    PreArgs {
                        x: src,
                        w: &dw,
                        scale: &dscale,
                        base: &dbase,
                        eps_norm,
                        eps_hc,
                        magnitude: mag,
                        n: N,
                        m,
                        mix,
                        y,
                        fault: cx.gpu.unlabelled_sink(),
                    },
                )
            };
        pre(&mut mix, &mut y, &dx, M)?;
        cx.hc.enqueue_head(
            st,
            HeadArgs {
                x: &dx,
                w: &dwh,
                scale: &dscale_head,
                base: &dbase_head,
                eps_norm,
                eps_hc,
                n: N,
                m: M,
                y: &mut yh,
                fault: cx.gpu.unlabelled_sink(),
            },
        )?;
        st.synchronize()?;
        let (mix_h, y_h, yh_h) = (
            mix.to_host_vec(st)?,
            y.to_host_vec(st)?,
            yh.to_host_vec(st)?,
        );
        let clean = some_fault(cx)?.is_none();

        // The gates against the rule in f64 (bound clause).
        let (mut want, mut bound) = (vec![], vec![]);
        let mut want_head = vec![];
        let mut bound_head = vec![];
        for t in 0..M {
            let xt = &x[t * K..(t + 1) * K];
            let r = mix_ref(
                xt,
                &w,
                MIX,
                &[f64::from(scale_h[0]), f64::from(scale_h[1])],
                &base_h,
                f64::from(eps_norm),
            );
            for row in 0..MIX {
                let m = if row < HC { 1.0 } else { f64::from(mag) };
                let (g, e) = gate_ref(r.v[row], r.e_v[row], m, f64::from(eps_hc));
                want.push(g);
                bound.push(e * SECOND_ORDER);
            }
            // The head: four rows, one scale, the pre gates only; its fold's
            // value is checked against the f64 fold of these gates.
            let h = mix_ref(
                xt,
                &wh,
                HC,
                &[f64::from(scale_head[0]), f64::from(scale_head[0])],
                &base_head,
                f64::from(eps_norm),
            );
            let gates: Vec<(f64, f64)> = (0..HC)
                .map(|i| gate_ref(h.v[i], h.e_v[i], 1.0, f64::from(eps_hc)))
                .collect();
            for d in 0..N {
                let mut v = 0.0;
                let mut e = 0.0;
                for (i, &(p, ep)) in gates.iter().enumerate() {
                    let xi = f64::from(xt[i * N + d]);
                    v += xi * p;
                    // The pre gate's own error, and the fold's: four products
                    // and three adds, each u of the term or partial sum.
                    e += xi.abs() * ep + 4.0 * U * (xi * p).abs();
                }
                want_head.push(v);
                bound_head.push(e * SECOND_ORDER);
            }
        }
        let (gates_ok, gates_worst) = within(&mix_h, &want, &bound);
        let mut ok = clause(
            "hc_pre gates against f64",
            gates_ok && clean,
            &format!("tokens={M} worst/bound={gates_worst:.3} clean={clean}"),
        );
        let (head_ok, head_worst) = within(&yh_h, &want_head, &bound_head);
        ok &= clause(
            "hc_head fold against f64",
            head_ok,
            &format!("worst/bound={head_worst:.3}"),
        );

        // The fold on the device's own pre gates: ggml's mul-then-add loop in
        // f32, bit for bit.
        let mut fold = vec![0.0f32; M * N];
        for t in 0..M {
            let p = &mix_h[t * MIX..t * MIX + HC];
            for d in 0..N {
                let xs = |i: usize| x[t * K + i * N + d];
                let a = xs(0) * p[0] + xs(1) * p[1];
                fold[t * N + d] = (a + xs(2) * p[2]) + xs(3) * p[3];
            }
        }
        ok &= clause(
            "hc_pre fold bit for bit",
            bits_equal(&y_h, &fold),
            "y = ((x0*p0 + x1*p1) + x2*p2) + x3*p3 over the device gates",
        );

        // HC post: out_i = res_i + x·post_i, bit for bit.
        let res = rng.fill(M * K, 1.0);
        let sub = rng.fill(M * N, 1.0);
        let (dres, dsub) = (
            DeviceBuffer::from_host(st, &res)?,
            DeviceBuffer::from_host(st, &sub)?,
        );
        let mut out = DeviceBuffer::zeroed(st, M * K)?;
        cx.hc.enqueue_post(
            st,
            PostArgs {
                x: &dsub,
                res: &dres,
                mix: &mix,
                n: N,
                m: M,
                out: &mut out,
            },
        )?;
        st.synchronize()?;
        let mut post = vec![0.0f32; M * K];
        for t in 0..M {
            for i in 0..HC {
                let p = mix_h[t * MIX + HC + i];
                for d in 0..N {
                    let at = (t * HC + i) * N + d;
                    post[at] = res[at] + sub[t * N + d] * p;
                }
            }
        }
        ok &= clause(
            "hc_post bit for bit",
            bits_equal(&out.to_host_vec(st)?, &post),
            "out = res + x*post",
        );

        // A non-finite stream value has no norm: the fault word names it,
        // that token's outputs are NaN and the other tokens do not move.
        let mut bad = x.clone();
        bad[K + 5] = f32::NAN;
        let dbad = DeviceBuffer::from_host(st, &bad)?;
        let mut mix2 = DeviceBuffer::zeroed(st, M * MIX)?;
        let mut y2 = DeviceBuffer::zeroed(st, M * N)?;
        pre(&mut mix2, &mut y2, &dbad, M)?;
        st.synchronize()?;
        let fault = some_fault(cx)?;
        let y2h = y2.to_host_vec(st)?;
        let mix2h = mix2.to_host_vec(st)?;
        let spared = bits_equal(&y2h[..N], &y_h[..N])
            && bits_equal(&y2h[2 * N..], &y_h[2 * N..])
            && bits_equal(&mix2h[..MIX], &mix_h[..MIX]);
        let poisoned = y2h[N..2 * N].iter().all(|v| v.is_nan());
        ok &= clause(
            "hc_pre refuses a NaN stream",
            fault == norm_fault() && poisoned && spared,
            &format!("fault={fault:?} poisoned={poisoned} others_unmoved={spared}"),
        );

        // One captured launch replays the eager run bit for bit.
        let graph = cx.gpu.capture(|s| {
            cx.hc.enqueue_pre(
                s,
                PreArgs {
                    x: &dx,
                    w: &dw,
                    scale: &dscale,
                    base: &dbase,
                    eps_norm,
                    eps_hc,
                    magnitude: mag,
                    n: N,
                    m: M,
                    mix: &mut mix2,
                    y: &mut y2,
                    fault: cx.gpu.unlabelled_sink(),
                },
            )
        })?;
        let mut replay = graph.node_count() == 1;
        for _ in 0..2 {
            graph.launch(st)?;
            st.synchronize()?;
            replay &= bits_equal(&y2.to_host_vec(st)?, &y_h)
                && bits_equal(&mix2.to_host_vec(st)?, &mix_h);
        }
        replay &= some_fault(cx)?.is_none();
        ok &= clause("hc_pre capture", replay, "one node, two exact replays");

        // Host launchers refuse a short buffer by name.
        let mut short = DeviceBuffer::zeroed(st, M * MIX - 1)?;
        let e = pre(&mut short, &mut y, &dx, M);
        let named =
            matches!(e, Err(GpuError::Shape { what, .. }) if what == "hc_ident::enqueue_pre");
        ok &= clause(
            "hc_pre host boundary",
            named,
            "a short mix is a named Shape error",
        );
        Ok(ok)
    }

    // ------------------------------------------------------------ engram

    fn engram_clauses(cx: &Cx) -> Result<bool, GateError> {
        const M: usize = 4;
        let st = cx.gpu.stream();
        let mut rng = Rng(0x00e9_9a6d_c0ff_ee11);
        let eps = bloomery_gpu::engram_delta::EPS;
        let mut ok = true;
        cx.gpu.take_fault()?;

        // Delta 1: the looked-up rows, at the scale a scale-less F8 row has.
        let rows = rng.fill(M * COLS * WIDTH, 200.0);
        let lookup: Vec<f32> = rng
            .fill(COLS * WIDTH, 1.0)
            .iter()
            .map(|v| v + 1.25)
            .collect();
        let mut valid = vec![1.0f32; M * COLS];
        valid[COLS + 3] = 0.0; // token 1 loses one column
        valid[2 * COLS..3 * COLS].fill(0.0); // token 2 has no valid column
        let (drows, dlookup, dvalid) = (
            DeviceBuffer::from_host(st, &rows)?,
            DeviceBuffer::from_host(st, &lookup)?,
            DeviceBuffer::from_host(st, &valid)?,
        );
        let mut rnorm = DeviceBuffer::zeroed(st, M * COLS * WIDTH)?;
        macro_rules! rows_args {
            ($rows:expr, $out:expr) => {
                RowsArgs {
                    rows: $rows,
                    lookup: &dlookup,
                    valid: &dvalid,
                    eps,
                    width: WIDTH,
                    cols: COLS,
                    m: M,
                    out: $out,
                    fault: cx.gpu.unlabelled_sink(),
                }
            };
        }
        cx.eng
            .enqueue_rows_norm(st, rows_args!(&drows, &mut rnorm))?;
        st.synchronize()?;
        let got = rnorm.to_host_vec(st)?;
        // The kernel: a lane's three fused multiply-adds, five butterfly
        // levels (R = 8 roundings on the squares), the mean's divide, eps's
        // add, sqrt and divide, then three products.
        let scale_rel = ((8.0 + 2.0) / 2.0 + 2.0) * U;
        let (mut want, mut bound) = (vec![], vec![]);
        for r in 0..M * COLS {
            let row = &rows[r * WIDTH..(r + 1) * WIDTH];
            let sq: f64 = row.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
            let scale = 1.0 / (sq / WIDTH as f64 + f64::from(eps)).sqrt();
            for (i, &v) in row.iter().enumerate() {
                let y = f64::from(v)
                    * scale
                    * f64::from(lookup[(r % COLS) * WIDTH + i])
                    * f64::from(valid[r]);
                want.push(y);
                bound.push(y.abs() * (scale_rel + 3.0 * U) * SECOND_ORDER);
            }
        }
        let (rows_ok, rows_worst) = within(&got, &want, &bound);
        let masked_exact = got[2 * COLS * WIDTH..3 * COLS * WIDTH]
            .iter()
            .all(|v| *v == 0.0);
        ok &= clause(
            "engram_rows_norm against f64",
            rows_ok && masked_exact && some_fault(cx)?.is_none(),
            &format!("worst/bound={rows_worst:.3} invalid_columns_exactly_zero={masked_exact}"),
        );

        // Delta 2: the bias add, bit for bit.
        let kv_raw = rng.fill(M * (HC + 1) * N, 4.0);
        let bias = rng.fill((HC + 1) * N, 2.0);
        let dbias = DeviceBuffer::from_host(st, &bias)?;
        let mut kv = DeviceBuffer::from_host(st, &kv_raw)?;
        cx.eng.enqueue_kv_bias(st, &dbias, M, &mut kv)?;
        st.synchronize()?;
        let kv_h = kv.to_host_vec(st)?;
        let sum: Vec<f32> = kv_raw
            .iter()
            .enumerate()
            .map(|(i, &v)| v + bias[i % ((HC + 1) * N)])
            .collect();
        ok &= clause(
            "engram_kv_bias bit for bit",
            bits_equal(&kv_h, &sum),
            "kv += bias",
        );

        // The key side at unit gains: kn = (k·scale)·1. The kernel's squares:
        // N/256 fused multiply-adds per thread, five butterfly and three tree
        // levels, so R = 12 + 8 = 20 roundings.
        let ones = DeviceBuffer::from_host(st, &vec![1.0f32; HC * N])?;
        let mut kn = DeviceBuffer::zeroed(st, M * HC * N)?;
        cx.eng.gate.enqueue_key_norm(
            st,
            KeyNormArgs {
                kv: &kv,
                gain: &ones,
                eps,
                hc: HC,
                m: M,
                kn: &mut kn,
                fault: cx.gpu.unlabelled_sink(),
            },
        )?;
        st.synchronize()?;
        let kn_h = kn.to_host_vec(st)?;
        let r20 = (N / 256 + 8) as f64;
        let scale20_rel = ((r20 + 2.0) / 2.0 + 2.0) * U;
        let norm_of = |row: &[f32]| -> f64 {
            let sq: f64 = row.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
            1.0 / (sq / N as f64 + f64::from(eps)).sqrt()
        };
        let (mut kwant, mut kbound) = (vec![], vec![]);
        for t in 0..M {
            for s in 0..HC {
                let row = &kv_h[(t * (HC + 1) + s) * N..(t * (HC + 1) + s + 1) * N];
                let sc = norm_of(row);
                for &v in row {
                    let y = f64::from(v) * sc;
                    kwant.push(y);
                    kbound.push(y.abs() * (scale20_rel + 2.0 * U) * SECOND_ORDER);
                }
            }
        }
        let (kn_ok, kn_worst) = within(&kn_h, &kwant, &kbound);
        ok &= clause(
            "engram_key_norm_3072 against f64",
            kn_ok,
            &format!("worst/bound={kn_worst:.3}"),
        );

        // The gate: streams whose key is aligned, anti-aligned, random, and a
        // zero stream (sign(0) = +1 as llama.cpp's graph takes it).
        let mut x = rng.fill(M * HC * N, 1.0);
        let mut kvg = kv_h;
        let key_at = |t: usize, s: usize| (t * (HC + 1) + s) * N;
        for d in 0..N {
            kvg[key_at(0, 0) + d] = x[d];
            kvg[key_at(0, 1) + d] = -x[N + d];
        }
        x[(HC + 1) * N..(HC + 2) * N].fill(0.0); // token 1, stream 1
        let egate: Vec<f32> = rng
            .fill(HC * N, 1.0)
            .iter()
            .map(|v| v * 0.5 + 0.25)
            .collect();
        let (dx, dgain, dkv) = (
            DeviceBuffer::from_host(st, &x)?,
            DeviceBuffer::from_host(st, &egate)?,
            DeviceBuffer::from_host(st, &kvg)?,
        );
        let mut kn2 = DeviceBuffer::zeroed(st, M * HC * N)?;
        let mut gate = DeviceBuffer::zeroed(st, M * HC)?;
        macro_rules! key_args {
            ($kv:expr, $kn:expr) => {
                KeyNormArgs {
                    kv: $kv,
                    gain: &ones,
                    eps,
                    hc: HC,
                    m: M,
                    kn: $kn,
                    fault: cx.gpu.unlabelled_sink(),
                }
            };
        }
        macro_rules! gate_args {
            ($kn:expr, $gate:expr) => {
                GateOnlyArgs {
                    x: &dx,
                    kn: $kn,
                    gain: &dgain,
                    eps,
                    hc: HC,
                    m: M,
                    gate: $gate,
                    fault: cx.gpu.unlabelled_sink(),
                }
            };
        }
        cx.eng
            .gate
            .enqueue_key_norm(st, key_args!(&dkv, &mut kn2))?;
        cx.eng.gate.enqueue_gate(st, gate_args!(&kn2, &mut gate))?;
        st.synchronize()?;
        let gate_h = gate.to_host_vec(st)?;
        // Per (token, stream): the dot of the two normalized rows with the
        // gain in f64, `Σ|term|` for its error. Each term carries kn's
        // relative error (R = 20 squares roundings, two products), qn's the
        // same, and its own product's: 31u. The dot rounds to f32 (u), times
        // 1/√N (a sqrt, a divide and a multiply: 3u), the sign and the root
        // (one rounding), and σ is 1/4-Lipschitz with three roundings.
        let inv = 1.0 / (N as f64).sqrt();
        let term_rel = (2.0 * (scale20_rel / U + 2.0) + 1.0) * U;
        let mut gwant = vec![];
        let mut gbound = vec![];
        let mut zero_gate = f64::NAN;
        for t in 0..M {
            for s in 0..HC {
                let xr = &x[(t * HC + s) * N..(t * HC + s + 1) * N];
                let kr = &kvg[key_at(t, s)..key_at(t, s) + N];
                let (sx, sk) = (norm_of(xr), norm_of(kr));
                let (mut dot, mut abs) = (0.0f64, 0.0f64);
                for d in 0..N {
                    let term =
                        f64::from(xr[d]) * sx * f64::from(kr[d]) * sk * f64::from(egate[s * N + d]);
                    dot += term;
                    abs += term.abs();
                }
                let sv = dot * inv;
                let e_s = term_rel * abs * inv + 4.0 * U * sv.abs();
                let (m, e_m) = if sv.abs() > 1e-6 {
                    let m = sv.signum() * sv.abs().sqrt();
                    (m, m.abs() * (e_s / (2.0 * sv.abs()) + U))
                } else {
                    let m = 1e-6f64.sqrt();
                    (m, m * U)
                };
                let g = sigmoid(m);
                if t == 1 && s == 1 {
                    zero_gate = g;
                }
                gwant.push(g);
                gbound.push((g * (1.0 - g) * e_m + 4.0 * U) * SECOND_ORDER);
            }
        }
        let (g_ok, g_worst) = within(&gate_h, &gwant, &gbound);
        ok &= clause(
            "engram_gate_3072 against f64",
            g_ok,
            &format!("streams={} worst/bound={g_worst:.3}", M * HC),
        );
        let zero_ok =
            (f64::from(gate_h[HC + 1]) - zero_gate).abs() <= 4.0 * U && gate_h[HC + 1] > 0.5;
        ok &= clause(
            "engram_gate_3072 sign(0) is +1",
            zero_ok,
            &format!("gate={} want=σ(1e-3)={zero_gate}", gate_h[HC + 1]),
        );

        // Deltas 3 and 4 and the mask, on the device's own gate and the
        // biased kv. The value's norm has R = 20 roundings, so its scale is
        // within `scale20_rel`; two products make the signal 15u; the sum
        // `gate + hc_scale`, the product with it, and the final add are one
        // rounding each (the mask's product is exact).
        let gamma = rng.fill(N, 1.0);
        let hcs = rng.fill(HC * N, 0.5);
        let (dgamma, dhcs) = (
            DeviceBuffer::from_host(st, &gamma)?,
            DeviceBuffer::from_host(st, &hcs)?,
        );
        let mut out = DeviceBuffer::zeroed(st, M * HC * N)?;
        macro_rules! delta_args {
            ($kv:expr, $gate:expr, $out:expr) => {
                DeltaArgs {
                    x: &dx,
                    kv: $kv,
                    gate: $gate,
                    gamma: &dgamma,
                    hc_scale: &dhcs,
                    valid: &dvalid,
                    eps,
                    cols: COLS,
                    m: M,
                    out: $out,
                    fault: cx.gpu.unlabelled_sink(),
                }
            };
        }
        cx.eng
            .enqueue_delta(st, delta_args!(&dkv, &gate, &mut out))?;
        st.synchronize()?;
        let out_h = out.to_host_vec(st)?;
        let (mut owant, mut obound) = (vec![], vec![]);
        for t in 0..M {
            let value = &kvg[key_at(t, HC)..key_at(t, HC) + N];
            let sv = norm_of(value);
            let any = if valid[t * COLS..(t + 1) * COLS].iter().sum::<f32>() > 0.0 {
                1.0
            } else {
                0.0
            };
            for s in 0..HC {
                let g = f64::from(gate_h[t * HC + s]);
                for d in 0..N {
                    let signal = f64::from(value[d]) * sv * f64::from(gamma[d]);
                    let delta = signal * (g + f64::from(hcs[s * N + d])) * any;
                    let xv = f64::from(x[(t * HC + s) * N + d]);
                    let o = xv + delta;
                    owant.push(o);
                    obound
                        .push((delta.abs() * (scale20_rel + 4.0 * U) + U * o.abs()) * SECOND_ORDER);
                }
            }
        }
        let (o_ok, o_worst) = within(&out_h, &owant, &obound);
        ok &= clause(
            "engram_delta_apply against f64",
            o_ok,
            &format!("worst/bound={o_worst:.3}"),
        );
        let masked = bits_equal(&out_h[2 * HC * N..3 * HC * N], &x[2 * HC * N..3 * HC * N]);
        let live = out_h[..HC * N] != x[..HC * N];
        ok &= clause(
            "engram_delta_apply masks a token with no valid column",
            masked && live,
            &format!("token2_out_equals_x={masked} token0_moved={live}"),
        );

        // A value row with no f32 norm: the fault word, that token's four
        // streams NaN, the other tokens unmoved.
        let mut kvbad = kvg.clone();
        kvbad[key_at(3, HC) + 7] = 1e20;
        let dkvbad = DeviceBuffer::from_host(st, &kvbad)?;
        let mut out2 = DeviceBuffer::zeroed(st, M * HC * N)?;
        cx.eng
            .enqueue_delta(st, delta_args!(&dkvbad, &gate, &mut out2))?;
        st.synchronize()?;
        let fault = some_fault(cx)?;
        let o2 = out2.to_host_vec(st)?;
        let spared = bits_equal(&o2[..3 * HC * N], &out_h[..3 * HC * N]);
        let poisoned = o2[3 * HC * N..].iter().all(|v| v.is_nan());
        ok &= clause(
            "engram_delta_apply refuses a value row with no norm",
            fault == norm_fault() && spared && poisoned,
            &format!("fault={fault:?} others_unmoved={spared} poisoned={poisoned}"),
        );
        let mut rbad = rows;
        rbad[5] = 1e20;
        let drbad = DeviceBuffer::from_host(st, &rbad)?;
        let mut rn2 = DeviceBuffer::zeroed(st, M * COLS * WIDTH)?;
        cx.eng.enqueue_rows_norm(st, rows_args!(&drbad, &mut rn2))?;
        st.synchronize()?;
        let fault = some_fault(cx)?;
        let r2 = rn2.to_host_vec(st)?;
        ok &= clause(
            "engram_rows_norm refuses a row with no norm",
            fault == norm_fault()
                && r2[..WIDTH].iter().all(|v| v.is_nan())
                && bits_equal(&r2[WIDTH..], &got[WIDTH..]),
            &format!("fault={fault:?}"),
        );

        // The chain as one graph (rows norm, key, gate, delta) replays the
        // eager run bit for bit.
        let graph = cx.gpu.capture(|s| {
            cx.eng.enqueue_rows_norm(s, rows_args!(&drows, &mut rn2))?;
            cx.eng.gate.enqueue_key_norm(s, key_args!(&dkv, &mut kn2))?;
            cx.eng.gate.enqueue_gate(s, gate_args!(&kn2, &mut gate))?;
            cx.eng.enqueue_delta(s, delta_args!(&dkv, &gate, &mut out2))
        })?;
        let mut replay = graph.node_count() == 4;
        for _ in 0..2 {
            graph.launch(st)?;
            st.synchronize()?;
            replay &= bits_equal(&out2.to_host_vec(st)?, &out_h)
                && bits_equal(&gate.to_host_vec(st)?, &gate_h)
                && bits_equal(&rn2.to_host_vec(st)?, &got);
        }
        replay &= some_fault(cx)?.is_none();
        ok &= clause(
            "engram chain capture",
            replay,
            "four nodes, two exact replays",
        );

        // Host launchers refuse a short buffer by name.
        let mut short = DeviceBuffer::zeroed(st, M * HC * N - 1)?;
        let e = cx
            .eng
            .enqueue_delta(st, delta_args!(&dkv, &gate, &mut short));
        let named =
            matches!(e, Err(GpuError::Shape { what, .. }) if what == "engram_delta::enqueue_delta");
        let mut small = DeviceBuffer::zeroed(st, M * (HC + 1) * N - 1)?;
        let e2 = cx.eng.enqueue_kv_bias(st, &dbias, M, &mut small);
        let named2 = matches!(e2, Err(GpuError::Shape { what, .. }) if what == "engram_delta::enqueue_kv_bias");
        ok &= clause(
            "engram host boundaries",
            named && named2,
            "a short out and a short kv are named Shape errors",
        );
        Ok(ok)
    }

    // ------------------------------------------------------------ router

    #[derive(Default)]
    struct Pick {
        scores: Vec<f32>,
        ids: Vec<u32>,
        weights: Vec<f32>,
    }

    impl Pick {
        fn same(&self, other: &Self) -> bool {
            bits_equal(&self.scores, &other.scores)
                && self.ids == other.ids
                && bits_equal(&self.weights, &other.weights)
        }
    }

    /// The top `k` keys, a tie to the higher id.
    fn top_k(key: &[f32], k: usize) -> Vec<u32> {
        let mut ids: Vec<u32> = (0..EXPERTS as u32).collect();
        ids.sort_by(|&a, &b| key[b as usize].total_cmp(&key[a as usize]).then(b.cmp(&a)));
        ids.truncate(k);
        ids
    }

    /// llama.cpp's `build_moe_ffn` where the file sets no
    /// `expert_weights_norm_eps`: the kept scores' sum, in f64, narrowed to
    /// f32 and floored at 2^-14; each weight `score / sum · scale`.
    fn weights_of(scores: &[f32], ids: &[u32], scale: f32) -> Vec<f32> {
        let sum = ids
            .iter()
            .fold(0.0f64, |sum, &e| sum + f64::from(scores[e as usize]));
        let div = (sum as f32).max(FLOOR);
        ids.iter()
            .map(|&e| scores[e as usize] / div * scale)
            .collect()
    }

    fn matrix(logits: &[f32]) -> Vec<f32> {
        let mut w = vec![0.0; EXPERTS * ROUTER_K];
        for (e, &logit) in logits.iter().enumerate() {
            w[e * ROUTER_K] = logit;
        }
        w
    }

    fn read_one(cx: &Cx, out: &RouterOut) -> Result<Pick, GateError> {
        let s = cx.gpu.stream();
        Ok(Pick {
            scores: out.probs.to_host_vec(s)?,
            ids: out.ids.to_host_vec(s)?,
            weights: out.weights.to_host_vec(s)?,
        })
    }

    fn router_case(cx: &Cx, name: &str, logits: &[f32], bias: &[f32]) -> Result<bool, GateError> {
        const TOKENS: usize = 3;
        let s = cx.gpu.stream();
        let w = DeviceTensor::upload(s, &matrix(logits), EXPERTS, ROUTER_K)?;
        let factors = [1.0f32, 0.0, 2.0];
        let mut xh = vec![0.0; TOKENS * ROUTER_K];
        for (t, &f) in factors.iter().enumerate() {
            xh[t * ROUTER_K] = f;
        }
        let x = DeviceBuffer::from_host(s, &xh)?;
        let b = DeviceBuffer::from_host(s, bias)?;
        let input = RouteIn {
            w: &w,
            bias: Some(&b),
            scale: SCALE,
            fault: cx.gpu.unlabelled_sink(),
        };
        let mut out = RouterOut::new(s)?;
        let mut single = Pick::default();
        let mut first = Pick::default();
        let mut tickets = true;
        cx.gpu.take_fault()?;
        for t in 0..TOKENS {
            let xt = DeviceBuffer::from_host(s, &xh[t * ROUTER_K..(t + 1) * ROUTER_K])?;
            cx.router.route(s, &xt, &input, &mut out)?;
            s.synchronize()?;
            let pick = read_one(cx, &out)?;
            tickets &= out.tickets(s)? == 0;
            if t == 0 {
                first = read_one(cx, &out)?;
            }
            single.scores.extend(pick.scores);
            single.ids.extend(pick.ids);
            single.weights.extend(pick.weights);
        }
        let mut probs = DeviceBuffer::zeroed(s, EXPERTS * TOKENS)?;
        let mut ids = DeviceBuffer::zeroed(s, USED * TOKENS)?;
        let mut weights = DeviceBuffer::zeroed(s, USED * TOKENS)?;
        macro_rules! rows {
            ($st:expr, $probs:expr, $ids:expr, $weights:expr) => {
                cx.router.route_rows(
                    ($st, &cx.gpu, &cx.wide),
                    &x,
                    &input,
                    TOKENS,
                    RoutedRows {
                        probs: $probs,
                        logits: None,
                        ids: $ids,
                        weights: $weights,
                    },
                )
            };
        }
        rows!(s, &mut probs, &mut ids, &mut weights)?;
        s.synchronize()?;
        let batch = Pick {
            scores: probs.to_host_vec(s)?,
            ids: ids.to_host_vec(s)?,
            weights: weights.to_host_vec(s)?,
        };
        let clean = some_fault(cx)?.is_none();
        let mut expected = true;
        for t in 0..TOKENS {
            let scores = &single.scores[t * EXPERTS..(t + 1) * EXPERTS];
            let keys: Vec<f32> = scores.iter().zip(bias).map(|(&p, &b)| p + b).collect();
            let want_ids = top_k(&keys, USED);
            expected &= single.ids[t * USED..(t + 1) * USED] == want_ids;
            expected &= bits_equal(
                &single.weights[t * USED..(t + 1) * USED],
                &weights_of(scores, &want_ids, SCALE),
            );
        }
        let mut ok = clause(
            &format!("router {name}: biased selection, unbiased floor weights"),
            expected && clean && tickets,
            &format!("clean={clean} tickets_zero={tickets}"),
        );
        ok &= clause(
            &format!("router {name}: rows are the single launches"),
            single.same(&batch),
            "bit_identical",
        );
        if name == "tiny" {
            let sum: f32 = first.ids.iter().map(|&e| first.scores[e as usize]).sum();
            let legacy: Vec<f32> = first
                .ids
                .iter()
                .map(|&e| first.scores[e as usize] / (sum + 1e-20) * SCALE)
                .collect();
            ok &= clause(
                "router tiny: the 2^-14 floor is active, not 1e-20",
                sum > 0.0
                    && sum < FLOOR
                    && !bits_equal(&first.weights, &legacy)
                    && first.weights.iter().sum::<f32>() < SCALE,
                "weights differ from the epsilon renormalization",
            );
        }
        if name == "ties" {
            let want: Vec<u32> = (0..USED as u32).map(|i| EXPERTS as u32 - 1 - i).collect();
            ok &= clause(
                "router ties: descending ids",
                first.ids == want,
                "sixteen ids",
            );
        }
        if name == "biased" {
            let top = top_k(&first.scores, USED);
            let keys: Vec<f32> = first
                .scores
                .iter()
                .zip(bias)
                .map(|(&p, &b)| p + b)
                .collect();
            ok &= clause(
                "router biased: the bias moves the pick, not the weights",
                first.ids != top
                    && !bits_equal(&first.weights, &weights_of(&keys, &first.ids, SCALE)),
                "unbiased kept scores are renormalized",
            );
        }
        let x1 = DeviceBuffer::from_host(s, &xh[..ROUTER_K])?;
        let graph = cx
            .gpu
            .capture(|st| cx.router.route(st, &x1, &input, &mut out))?;
        let mut replay = graph.node_count() == 1;
        for _ in 0..2 {
            graph.launch(s)?;
            s.synchronize()?;
            replay &= read_one(cx, &out)?.same(&first) && out.tickets(s)? == 0;
        }
        let graph = cx
            .gpu
            .capture(|st| rows!(st, &mut probs, &mut ids, &mut weights))?;
        let mut rows_replay = graph.node_count() == 3;
        for _ in 0..2 {
            graph.launch(s)?;
            s.synchronize()?;
            let now = Pick {
                scores: probs.to_host_vec(s)?,
                ids: ids.to_host_vec(s)?,
                weights: weights.to_host_vec(s)?,
            };
            rows_replay &= now.same(&single);
        }
        rows_replay &= some_fault(cx)?.is_none();
        ok &= clause(
            &format!("router {name}: capture"),
            replay && rows_replay,
            "one node and three nodes, exact replays, no fault",
        );
        Ok(ok)
    }

    fn router_nan(cx: &Cx) -> Result<bool, GateError> {
        let s = cx.gpu.stream();
        let mut logits = vec![0.0; EXPERTS];
        logits[40] = f32::NAN;
        let w = DeviceTensor::upload(s, &matrix(&logits), EXPERTS, ROUTER_K)?;
        let b = DeviceBuffer::zeroed(s, EXPERTS)?;
        let mut xh = vec![0.0; ROUTER_K];
        xh[0] = 1.0;
        let x = DeviceBuffer::from_host(s, &xh)?;
        let input = RouteIn {
            w: &w,
            bias: Some(&b),
            scale: SCALE,
            fault: cx.gpu.unlabelled_sink(),
        };
        let mut out = RouterOut::new(s)?;
        cx.gpu.take_fault()?;
        cx.router.route(s, &x, &input, &mut out)?;
        s.synchronize()?;
        let fault = some_fault(cx)?;
        let poisoned = out.weights.to_host_vec(s)?.iter().all(|w| !w.is_finite());
        let tickets = out.tickets(s)?;
        let want = Some(Fault::at(LAYER_NONE, FaultSite::Router));
        // The bias is required: a call without one is a named error.
        let none = cx
            .router
            .route(
                s,
                &x,
                &RouteIn {
                    bias: None,
                    ..input
                },
                &mut out,
            )
            .is_err();
        Ok(clause(
            "router refuses a NaN logit and a missing bias",
            fault == want && poisoned && tickets == 0 && none,
            &format!(
                "fault={fault:?} poisoned={poisoned} tickets={tickets} no_bias_refused={none}"
            ),
        ))
    }

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let gpu = Gpu::new()?;
        println!("gate_hy4_hc: device {}", gpu.device_name()?);
        let ctx = gpu.context();
        let cx = Cx {
            hc: HcIdentKernels::load(ctx)?,
            eng: EngramDeltaKernels::load(ctx)?,
            router: RouterKernels::load(ctx)?,
            wide: Gemm32Kernels::load(ctx)?,
            gpu,
        };
        let mut ok = hc_clauses(&cx)?;
        ok &= engram_clauses(&cx)?;

        let normal: Vec<f32> = (0..EXPERTS)
            .map(|e| {
                if e < 32 {
                    2.0 - e as f32 * 0.125
                } else {
                    -20.0
                }
            })
            .collect();
        let tiny: Vec<f32> = (0..EXPERTS)
            .map(|e| -30.0 - (e % 32) as f32 * 0.25)
            .collect();
        let mut bias = vec![0.0f32; EXPERTS];
        for (e, b) in bias.iter_mut().enumerate().skip(EXPERTS - 24) {
            *b = 2.0 + e as f32 / EXPERTS as f32;
        }
        let none = vec![0.0f32; EXPERTS];
        ok &= router_case(&cx, "normal", &normal, &none)?;
        ok &= router_case(&cx, "biased", &normal, &bias)?;
        ok &= router_case(&cx, "tiny", &tiny, &none)?;
        ok &= router_case(&cx, "ties", &vec![0.0; EXPERTS], &none)?;
        ok &= router_nan(&cx)?;
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
