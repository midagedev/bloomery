//! The actual 256/top-6 selected-sum-floor binding, through RouterEntry.
//! Optional selection-only bias, active floor/zero sums, ties, exact rows,
//! nonfinite refusal and graph replay. This gate needs an NVIDIA device.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_flint_router", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::gemm::Gemm32Kernels;
    use bloomery_gpu::route_body::sigmoid_floor_256_6::{RouterKernels, RouterOut};
    use bloomery_gpu::route_core::sigmoid;
    use bloomery_gpu::routed_block::{RouteIn, RoutedRows, RouterEntry};
    use bloomery_gpu::{DeviceTensor, Fault, FaultSite, Gpu, LAYER_NONE};
    use bloomery_gpu_gates::{GateError, bits_equal, checks_failed, verdict};
    use cuda_core::DeviceBuffer;

    const EXPERTS: usize = 256;
    const USED: usize = 6;
    const K: usize = 64;
    const TOKENS: usize = 5;
    const SCALE: f32 = 1.75;
    const FLOOR: f32 = 1.0 / 16384.0;

    struct Cx {
        gpu: Gpu,
        router: RouterKernels,
        wide: Gemm32Kernels,
    }

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

    fn top_k(key: &[f32], k: usize) -> Vec<u32> {
        let mut ids: Vec<u32> = (0..EXPERTS as u32).collect();
        ids.sort_by(|&a, &b| key[b as usize].total_cmp(&key[a as usize]).then(b.cmp(&a)));
        ids.truncate(k);
        ids
    }

    // Independent contract expression: do not reuse the device divisor.
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
        let mut w = vec![0.0; EXPERTS * K];
        for (e, &logit) in logits.iter().enumerate() {
            w[e * K] = logit;
        }
        w
    }

    fn activations(factors: &[f32]) -> Vec<f32> {
        let mut x = vec![0.0; factors.len() * K];
        for (t, &factor) in factors.iter().enumerate() {
            x[t * K] = factor;
        }
        x
    }

    fn read_one(cx: &Cx, out: &RouterOut) -> Result<Pick, GateError> {
        let s = cx.gpu.stream();
        Ok(Pick {
            scores: out.probs.to_host_vec(s)?,
            ids: out.ids.to_host_vec(s)?,
            weights: out.weights.to_host_vec(s)?,
        })
    }

    fn clause(name: &str, pass: bool, detail: &str) -> bool {
        println!("{name} {detail} {}", verdict(pass));
        pass
    }

    fn valid_case(
        cx: &Cx,
        name: &str,
        logits: &[f32],
        bias: Option<&[f32]>,
    ) -> Result<bool, GateError> {
        let s = cx.gpu.stream();
        let mut wh = matrix(logits);
        if name == "dense" {
            for e in 0..EXPERTS {
                for j in 0..K {
                    wh[e * K + j] = ((e * 17 + j * 11) % 67) as f32 / 67.0 - 0.5;
                }
            }
        }
        let w = DeviceTensor::upload(s, &wh, EXPERTS, K)?;
        let factors = [1.0, 0.0, 2.0, -1.0, 0.5];
        let mut xh = activations(&factors);
        if name == "dense" {
            for t in 0..TOKENS {
                for j in 1..K {
                    xh[t * K + j] = ((t * 13 + j * 7) % 31) as f32 / 31.0 - 0.5;
                }
            }
        }
        let x = DeviceBuffer::from_host(s, &xh)?;
        let b = bias.map(|b| DeviceBuffer::from_host(s, b)).transpose()?;
        let input = RouteIn {
            w: &w,
            bias: b.as_ref(),
            scale: SCALE,
            fault: cx.gpu.unlabelled_sink(),
        };
        let mut out = RouterOut::new(s)?;
        let mut single = Pick::default();
        let mut first = Pick::default();
        let mut tickets = true;
        cx.gpu.take_fault()?;
        for t in 0..TOKENS {
            let xt = DeviceBuffer::from_host(s, &xh[t * K..(t + 1) * K])?;
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
        cx.router.route_rows(
            (s, &cx.gpu, &cx.wide),
            &x,
            &input,
            TOKENS,
            RoutedRows {
                probs: &mut probs,
                logits: None,
                ids: &mut ids,
                weights: &mut weights,
            },
        )?;
        s.synchronize()?;
        let batch = Pick {
            scores: probs.to_host_vec(s)?,
            ids: ids.to_host_vec(s)?,
            weights: weights.to_host_vec(s)?,
        };
        let clean = cx.gpu.take_fault()?.is_none();
        let mut expected = true;
        for (t, &factor) in factors.iter().enumerate() {
            let scores = &single.scores[t * EXPERTS..(t + 1) * EXPERTS];
            let keys: Vec<f32> = scores
                .iter()
                .enumerate()
                .map(|(e, &p)| bias.map_or(p, |b| p + b[e]))
                .collect();
            let want_ids = top_k(&keys, USED);
            expected &= single.ids[t * USED..(t + 1) * USED] == want_ids;
            if name != "dense" {
                let planted_keys: Vec<f32> = logits
                    .iter()
                    .enumerate()
                    .map(|(e, &logit)| {
                        let p = sigmoid(logit * factor);
                        bias.map_or(p, |b| p + b[e])
                    })
                    .collect();
                expected &= single.ids[t * USED..(t + 1) * USED] == top_k(&planted_keys, USED);
            }
            expected &= bits_equal(
                &single.weights[t * USED..(t + 1) * USED],
                &weights_of(scores, &want_ids, SCALE),
            );
        }
        let mut ok = clause(
            &format!("{name} selection and unbiased floor weights"),
            expected && clean && tickets,
            &format!(
                "bias={} clean={clean} tickets_zero={tickets}",
                bias.is_some()
            ),
        );
        ok &= clause(
            &format!("{name} token-major rows"),
            single.same(&batch),
            "bit_identical",
        );
        if name.starts_with("tiny") {
            let sum: f32 = first.ids.iter().map(|&e| first.scores[e as usize]).sum();
            let legacy: Vec<f32> = first
                .ids
                .iter()
                .map(|&e| first.scores[e as usize] / (sum + 1e-20) * SCALE)
                .collect();
            ok &= clause(
                "tiny floor active",
                sum > 0.0
                    && sum < FLOOR
                    && !bits_equal(&first.weights, &legacy)
                    && first.weights.iter().sum::<f32>() < SCALE,
                "not epsilon renormalization",
            );
        }
        if name == "zero" || name == "ties" {
            ok &= clause(
                &format!("{name} descending-id ties"),
                first.ids == [255, 254, 253, 252, 251, 250],
                "six distinct ids",
            );
            if name == "zero" {
                ok &= clause(
                    "zero selected sum",
                    first.weights.iter().all(|&w| w == 0.0),
                    "finite zero weights",
                );
            }
        }
        if let Some(bias) = bias {
            let keys: Vec<f32> = first
                .scores
                .iter()
                .zip(bias)
                .map(|(&p, &b)| p + b)
                .collect();
            ok &= clause(
                "bias changes selection not weight source",
                first.ids != top_k(&first.scores, USED)
                    && !bits_equal(&first.weights, &weights_of(&keys, &first.ids, SCALE)),
                "unbiased selected scores consumed",
            );
        } else if name == "normal" {
            let top7 = top_k(&first.scores, 7);
            ok &= clause(
                "six not seven selected scores",
                !bits_equal(
                    &first.weights,
                    &weights_of(&first.scores, &top7, SCALE)[..USED],
                ),
                "top-seven renormalization differs",
            );
        }

        let x1 = DeviceBuffer::from_host(s, &xh[..K])?;
        let graph = cx
            .gpu
            .capture(|st| cx.router.route(st, &x1, &input, &mut out))?;
        let mut replay = graph.node_count() == 1;
        for _ in 0..2 {
            graph.launch(s)?;
            s.synchronize()?;
            replay &= read_one(cx, &out)?.same(&first) && out.tickets(s)? == 0;
        }
        ok &= clause(
            &format!("{name} single capture"),
            replay,
            "one node, two exact replays, zero tickets",
        );
        let graph = cx.gpu.capture(|st| {
            cx.router.route_rows(
                (st, &cx.gpu, &cx.wide),
                &x,
                &input,
                TOKENS,
                RoutedRows {
                    probs: &mut probs,
                    logits: None,
                    ids: &mut ids,
                    weights: &mut weights,
                },
            )
        })?;
        let mut replay = graph.node_count() == 3;
        for _ in 0..2 {
            graph.launch(s)?;
            s.synchronize()?;
            let now = Pick {
                scores: probs.to_host_vec(s)?,
                ids: ids.to_host_vec(s)?,
                weights: weights.to_host_vec(s)?,
            };
            replay &= now.same(&single);
        }
        replay &= cx.gpu.take_fault()?.is_none();
        ok &= clause(
            &format!("{name} rows capture"),
            replay,
            "three nodes, two exact replays, no fault",
        );
        Ok(ok)
    }

    fn invalid_case(
        cx: &Cx,
        name: &str,
        logit: Option<f32>,
        bias: Option<f32>,
        scale: f32,
    ) -> Result<bool, GateError> {
        let s = cx.gpu.stream();
        let mut logits = vec![0.0; EXPERTS];
        if let Some(logit) = logit {
            logits[40] = logit;
        }
        let w = DeviceTensor::upload(s, &matrix(&logits), EXPERTS, K)?;
        let mut bh = vec![0.0; EXPERTS];
        if let Some(bias) = bias {
            bh[40] = bias;
        }
        let b = bias.map(|_| DeviceBuffer::from_host(s, &bh)).transpose()?;
        let x = DeviceBuffer::from_host(s, &activations(&[1.0, 1.0]))?;
        let input = RouteIn {
            w: &w,
            bias: b.as_ref(),
            scale,
            fault: cx.gpu.unlabelled_sink(),
        };
        let mut out = RouterOut::new(s)?;
        cx.gpu.take_fault()?;
        cx.router.route(s, &x, &input, &mut out)?;
        s.synchronize()?;
        let one = cx.gpu.take_fault()?;
        let one_poisoned = out.weights.to_host_vec(s)?.iter().all(|w| !w.is_finite());
        let tickets = out.tickets(s)?;
        let mut probs = DeviceBuffer::zeroed(s, EXPERTS * 2)?;
        let mut ids = DeviceBuffer::zeroed(s, USED * 2)?;
        let mut weights = DeviceBuffer::zeroed(s, USED * 2)?;
        cx.router.route_rows(
            (s, &cx.gpu, &cx.wide),
            &x,
            &input,
            2,
            RoutedRows {
                probs: &mut probs,
                logits: None,
                ids: &mut ids,
                weights: &mut weights,
            },
        )?;
        s.synchronize()?;
        let rows = cx.gpu.take_fault()?;
        let rows_poisoned = weights.to_host_vec(s)?.iter().all(|w| !w.is_finite());
        let want = Some(Fault::at(LAYER_NONE, FaultSite::Router));
        Ok(clause(
            name,
            one == want && rows == want && one_poisoned && rows_poisoned && tickets == 0,
            &format!(
                "one={one:?} rows={rows:?} poisoned={} tickets={tickets}",
                one_poisoned && rows_poisoned
            ),
        ))
    }

    fn host_boundaries(cx: &Cx) -> Result<bool, GateError> {
        let s = cx.gpu.stream();
        let w = DeviceTensor::upload(s, &matrix(&vec![0.0; EXPERTS]), EXPERTS, K)?;
        let x = DeviceBuffer::zeroed(s, K)?;
        let short_bias = DeviceBuffer::zeroed(s, EXPERTS - 1)?;
        let mut out = RouterOut::new(s)?;
        let sink = cx.gpu.unlabelled_sink();
        let short = cx
            .router
            .enqueue_router(s, &w, &x, Some(&short_bias), SCALE, &mut out, sink)
            .is_err();
        let mut probs = DeviceBuffer::zeroed(s, EXPERTS)?;
        let mut ids = DeviceBuffer::zeroed(s, USED)?;
        let mut weights = DeviceBuffer::zeroed(s, USED)?;
        let empty = cx
            .router
            .enqueue_router_rows(
                s,
                &cx.gpu,
                &cx.wide,
                &w,
                &x,
                None,
                SCALE,
                0,
                &mut probs,
                &mut ids,
                &mut weights,
                sink,
            )
            .is_err();
        let short_rows = cx
            .router
            .enqueue_router_rows(
                s,
                &cx.gpu,
                &cx.wide,
                &w,
                &x,
                None,
                SCALE,
                2,
                &mut probs,
                &mut ids,
                &mut weights,
                sink,
            )
            .is_err();
        let legacy =
            bloomery_gpu::route_body::sigmoid_256_6::RouterKernels::load(cx.gpu.context())?;
        let mut legacy_out = bloomery_gpu::route_body::sigmoid_256_6::RouterOut::new(s)?;
        let missing = legacy
            .route(
                s,
                &x,
                &RouteIn {
                    w: &w,
                    bias: None,
                    scale: SCALE,
                    fault: sink,
                },
                &mut legacy_out,
            )
            .is_err();
        Ok(clause(
            "host boundaries",
            short && empty && short_rows && missing,
            "short bias, empty/short rows, legacy missing bias refused",
        ))
    }

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let gpu = Gpu::new()?;
        println!("gate_flint_router: device {}", gpu.device_name()?);
        let router = RouterKernels::load(gpu.context())?;
        let wide = Gemm32Kernels::load(gpu.context())?;
        let cx = Cx { gpu, router, wide };
        let normal: Vec<f32> = (0..EXPERTS)
            .map(|e| if e < 16 { 2.0 - e as f32 * 0.25 } else { -20.0 })
            .collect();
        let tiny: Vec<f32> = (0..EXPERTS)
            .map(|e| -30.0 - (e % 16) as f32 * 0.25)
            .collect();
        let mut bias = vec![0.0; EXPERTS];
        for (e, b) in bias.iter_mut().enumerate().skip(240) {
            *b = 2.0 + e as f32 / 256.0;
        }
        let mut ok = valid_case(&cx, "normal", &normal, None)?;
        ok &= valid_case(&cx, "dense", &normal, None)?;
        ok &= valid_case(&cx, "biased", &normal, Some(&bias))?;
        ok &= valid_case(&cx, "tiny", &tiny, None)?;
        ok &= valid_case(&cx, "tiny-biased", &tiny, Some(&bias))?;
        ok &= valid_case(&cx, "zero", &vec![-200.0; EXPERTS], None)?;
        ok &= valid_case(&cx, "ties", &vec![0.0; EXPERTS], None)?;
        for (name, value) in [
            ("NaN", f32::NAN),
            ("+infinity", f32::INFINITY),
            ("-infinity", f32::NEG_INFINITY),
        ] {
            ok &= invalid_case(&cx, &format!("{name} logit"), Some(value), None, SCALE)?;
            ok &= invalid_case(
                &cx,
                &format!("{name} selection bias"),
                None,
                Some(value),
                SCALE,
            )?;
        }
        ok &= invalid_case(&cx, "nonfinite weight scale", None, None, f32::INFINITY)?;
        ok &= host_boundaries(&cx)?;
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
