//! GPU gate for the grouped-query flash decode (`flash_gqa`) against ik's
//! `fa-L` (FLASH_ATTN_EXT) on its own inputs: the query rows `q-L`, the
//! cache views `k-L`/`v-L` (f16, `[n_kv][keys][128]` in logical order —
//! the planes' own layout) and the mask's live prefix, at every layer of
//! the decode-step sets: 5, 1,025 and 4,097 visible keys.
//!
//! The kernel is not bit-identical to ik by construction (another
//! exponential, other sum orders), so the comparison is a derived band. Per
//! output value `o_d` of head `h`, with the exact attention computed here in
//! f64 on the same f16 keys and values and f32 query — normalized weights
//! `p̄_j`, `A_j = scale·Σ_d |q_d·k_jd|`, `X_j = M − s_j` — each side's
//! first-order distance to the exact value is at most
//!
//! `Σ_j p̄_j·(ε_s·A_j + ε_e(X_j) + ε_r)·|v_jd − o_d| + γ(n_acc)·(Σ_j p̄_j·|v_jd| + |o_d|) + 2u·|o_d|`
//!
//! - `ε_s`: the score dot's roundings — ours `γ(35)` (32 fused
//!   multiply-adds per partial, two combine levels, the scale), ik's
//!   `γ(129)` (any order over 128 products, and the scale);
//! - `ε_e(X) = 4u + 2u·X`: the exponential's own error and the rounding of
//!   its argument (`s − M`, then ours `· log2 e`);
//! - `ε_r = 4u·R + 2u·max X`: the running-max rescales, `R` of them at most
//!   (ours one per tile and per segment, ik's one per 32-key block and per
//!   thread chunk);
//! - `γ(n_acc)`: the value and weight sums — ours a segment's keys, the
//!   segments and the butterfly, ik's every key serially.
//!
//! The tensor-core pass (`gqa_flash_seg_mma`) rounds the query to f16 for
//! its scores, so its `ε_s` is `2^-11 + 2·γ(130)` (the f16 rounding, and
//! the tensor core's f32 accumulation, whose order is undocumented) on
//! magnitudes that include `2^-14·scale·Σ|k|` for query values below f16's
//! normal range; everything after the scores is the scalar pass's.
//!
//! The f16 rounding of K and V costs nothing here: both sides read the same
//! f16 bits. Asserted: ours within its bound of the exact value, ik within
//! its bound (the check that this model of ik is right), ours within the sum
//! of both of ik's `fa-L`; a rerun bit-identical; padded cache rows set to
//! NaN change no bit; the captured graph (two nodes) replays bit-identical.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen3moe_flash: built without the `gpu` feature; see `just gate-gpu-qwen3moe-flash`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen3moe_flash", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::Gpu;
    use bloomery_gpu::flash_gqa::{
        FlashGqaKernels, GROUP, GqaArgs, HEAD, KEY_TILE, SEG_KEYS, partials_ms_len, partials_v_len,
        segments_for,
    };
    use bloomery_gpu_gates::qwen3moe::{f16_logical_bits, step_sets};
    use bloomery_gpu_gates::rounding::{U, gamma};
    use bloomery_gpu_gates::{
        GateError, bits_equal, checks_failed, mask_bits_in, ref_tensor_logical_in, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::quant::half_to_f32;

    /// An f16 NaN the padded rows are overwritten with.
    const NAN16: u16 = 0x7e00;

    /// One layer's inputs on the card.
    struct Inputs {
        q: DeviceBuffer<f32>,
        kc: DeviceBuffer<u16>,
        vc: DeviceBuffer<u16>,
        n_keys: DeviceBuffer<u32>,
    }

    /// One launch into fresh scratch, read back.
    fn run_once(
        k: &FlashGqaKernels,
        stream: &CudaStream,
        inp: &Inputs,
        scale: f32,
        n_kv: usize,
        ctx: usize,
        mma: bool,
    ) -> Result<Vec<f32>, GateError> {
        let n_head = n_kv * GROUP;
        let mut pv = DeviceBuffer::<f32>::zeroed(stream, partials_v_len(n_head, ctx))?;
        let mut pms = DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(n_head, ctx))?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, n_head * HEAD)?;
        k.enqueue_pass(
            stream,
            GqaArgs {
                q: &inp.q,
                kc: &inp.kc,
                vc: &inp.vc,
                n_keys: &inp.n_keys,
                scale,
                n_kv,
                ctx,
                part_v: &mut pv,
                part_ms: &mut pms,
                y: &mut y,
            },
            mma,
        )?;
        stream.synchronize()?;
        Ok(y.to_host_vec(stream)?)
    }

    /// The exact attention of one head in f64, and each side's bound.
    struct Exact {
        o: Vec<f64>,
        bound_ours: Vec<f64>,
        bound_mma: Vec<f64>,
        bound_ik: Vec<f64>,
    }

    fn exact(q: &[f32], kh: &[f32], vh: &[f32], n: usize, scale: f32) -> Exact {
        let sc = f64::from(scale);
        let s: Vec<f64> = (0..n)
            .map(|j| {
                let k = &kh[j * HEAD..(j + 1) * HEAD];
                sc * q
                    .iter()
                    .zip(k)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum::<f64>()
            })
            .collect();
        let a: Vec<f64> = (0..n)
            .map(|j| {
                let k = &kh[j * HEAD..(j + 1) * HEAD];
                sc.abs()
                    * q.iter()
                        .zip(k)
                        .map(|(&a, &b)| (f64::from(a) * f64::from(b)).abs())
                        .sum::<f64>()
            })
            .collect();
        // The tensor-core pass's scores also carry the query's f16 rounding:
        // 2^-11 of a normal value, 2^-25 absolute below f16's normal range
        // (2^-14 of the key magnitudes then, relative to the 2^-11 term).
        let a16: Vec<f64> = (0..n)
            .map(|j| {
                let k = &kh[j * HEAD..(j + 1) * HEAD];
                a[j] + sc.abs()
                    * 2f64.powi(-14)
                    * k.iter().map(|&b| f64::from(b).abs()).sum::<f64>()
            })
            .collect();
        let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let p: Vec<f64> = s.iter().map(|&v| (v - m).exp()).collect();
        let z: f64 = p.iter().sum();
        let pb: Vec<f64> = p.iter().map(|&v| v / z).collect();
        let x: Vec<f64> = s.iter().map(|&v| m - v).collect();
        let xmax = x.iter().copied().fold(0.0f64, f64::max);
        let segs = n.div_ceil(SEG_KEYS);
        let r_ours = (n.div_ceil(KEY_TILE) + segs) as f64;
        let r_ik = (n.div_ceil(32) + 33) as f64;
        let (es_o, es_i) = (gamma(35), gamma(129));
        // The tensor core's f32 accumulation order is not documented and may
        // truncate: twice γ(130) for the dot of exact f16 products.
        let es_m = 2f64.powi(-11) + 2.0 * gamma(130);
        let (acc_o, acc_i) = (gamma(SEG_KEYS + segs + 8), gamma(n + 40));
        let mut o = vec![0.0f64; HEAD];
        for (j, &w) in pb.iter().enumerate() {
            for d in 0..HEAD {
                o[d] += w * f64::from(vh[j * HEAD + d]);
            }
        }
        let mut bo = vec![0.0f64; HEAD];
        let mut bm = vec![0.0f64; HEAD];
        let mut bi = vec![0.0f64; HEAD];
        for d in 0..HEAD {
            let (mut t_o, mut t_m, mut t_i, mut pv) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for j in 0..n {
                let v = f64::from(vh[j * HEAD + d]);
                let dev = (v - o[d]).abs();
                let ee = 4.0 * U + 2.0 * U * x[j];
                t_o += pb[j] * (es_o * a[j] + ee + 4.0 * U * r_ours + 2.0 * U * xmax) * dev;
                t_m += pb[j] * (es_m * a16[j] + ee + 4.0 * U * r_ours + 2.0 * U * xmax) * dev;
                t_i += pb[j] * (es_i * a[j] + ee + 4.0 * U * r_ik + 2.0 * U * xmax) * dev;
                pv += pb[j] * v.abs();
            }
            bo[d] = t_o + acc_o * (pv + o[d].abs()) + 2.0 * U * o[d].abs();
            bm[d] = t_m + acc_o * (pv + o[d].abs()) + 2.0 * U * o[d].abs();
            bi[d] = t_i + acc_i * (pv + o[d].abs()) + 2.0 * U * o[d].abs();
        }
        Exact {
            o,
            bound_ours: bo,
            bound_mma: bm,
            bound_ik: bi,
        }
    }

    /// One pass's measured-over-bound ratios over a set, and the plain
    /// relative distance to ik.
    #[derive(Default)]
    struct Ratios {
        ours: f64,
        ik: f64,
        pair: f64,
        plain: f64,
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let k = FlashGqaKernels::load(gpu.context())?;
        let stream = gpu.stream();
        let scale = 1.0f32 / (HEAD as f32).sqrt();
        println!(
            "gate_qwen3moe_flash: device {} — head {HEAD}, group {GROUP}, {SEG_KEYS}-key segments, scale {scale:e}",
            gpu.device_name()?
        );
        let mut ok = true;
        let mut graph_done = false;
        for (label, man) in step_sets()? {
            let mut rs = [Ratios::default(), Ratios::default()];
            let (mut n_live, mut ctx_seen, mut layers) = (0usize, 0usize, 0usize);
            let mut set_ok = true;
            let mrow = man.input("KQ_mask", 0)?;
            let mask = mask_bits_in(&man.dir, mrow)?;
            let width = mrow.ne[0] as usize;
            let row0 = &mask[..width];
            let live = row0.iter().take_while(|&&b| b == 0).count();
            if live == 0 || row0[live..].iter().any(|&b| b != 0xfc00) {
                return Err(format!("{label}: KQ_mask row 0 is not a visible prefix").into());
            }
            while let Ok(fa) = man.tensor(&format!("fa-{layers}"), 0) {
                let l = layers;
                let qrow = man.tensor(&format!("q-{l}"), 0)?;
                let krow = man.tensor(&format!("k-{l}"), 0)?;
                let vrow = man.tensor(&format!("v-{l}"), 0)?;
                let (ctx, n_kv) = (krow.ne[1] as usize, krow.ne[2] as usize);
                let n_head = n_kv * GROUP;
                if krow.ne[0] as usize != HEAD
                    || vrow.ne != krow.ne
                    || qrow.ne != [HEAD as u64, 1, n_head as u64, 1]
                    || fa.ne != [HEAD as u64, n_head as u64, 1, 1]
                    || ctx != width
                {
                    return Err(format!(
                        "{label} layer {l}: q {:?} k {:?} v {:?} fa {:?} mask width {width}",
                        qrow.ne, krow.ne, vrow.ne, fa.ne
                    )
                    .into());
                }
                let q = ref_tensor_logical_in(&man.dir, qrow)?;
                let kb = f16_logical_bits(&man.dir, krow)?;
                let vb = f16_logical_bits(&man.dir, vrow)?;
                let want = ref_tensor_logical_in(&man.dir, fa)?;
                let pad = |b: &[u16]| -> Vec<u16> {
                    b.iter()
                        .enumerate()
                        .map(|(i, &h)| if (i / HEAD) % ctx >= live { NAN16 } else { h })
                        .collect()
                };
                let up = |kc: &[u16], vc: &[u16]| -> Result<Inputs, GateError> {
                    Ok(Inputs {
                        q: DeviceBuffer::from_host(stream, &q)?,
                        kc: DeviceBuffer::from_host(stream, kc)?,
                        vc: DeviceBuffer::from_host(stream, vc)?,
                        n_keys: DeviceBuffer::from_host(stream, &[live as u32])?,
                    })
                };
                let inp = up(&kb, &vb)?;
                let inp_nan = up(&pad(&kb), &pad(&vb))?;
                let kf: Vec<f32> = kb.iter().map(|&h| half_to_f32(h)).collect();
                let vf: Vec<f32> = vb.iter().map(|&h| half_to_f32(h)).collect();
                let exacts: Vec<Exact> = (0..n_head)
                    .map(|h| {
                        let plane = (h / GROUP) * ctx * HEAD;
                        exact(
                            &q[h * HEAD..(h + 1) * HEAD],
                            &kf[plane..plane + live * HEAD],
                            &vf[plane..plane + live * HEAD],
                            live,
                            scale,
                        )
                    })
                    .collect();
                let mx_ik = want.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                let mut ys = Vec::new();
                for (pass, mma) in [false, true].into_iter().enumerate() {
                    let y = run_once(&k, stream, &inp, scale, n_kv, ctx, mma)?;
                    let y2 = run_once(&k, stream, &inp, scale, n_kv, ctx, mma)?;
                    let y_nan = run_once(&k, stream, &inp_nan, scale, n_kv, ctx, mma)?;
                    let rerun = bits_equal(&y, &y2);
                    let nan_same = bits_equal(&y, &y_nan);
                    let mut layer_ok = rerun && nan_same;
                    let r = &mut rs[pass];
                    for (h, ex) in exacts.iter().enumerate() {
                        for d in 0..HEAD {
                            let (o, iv) =
                                (f64::from(y[h * HEAD + d]), f64::from(want[h * HEAD + d]));
                            let e = ex.o[d];
                            let bo = if mma {
                                ex.bound_mma[d]
                            } else {
                                ex.bound_ours[d]
                            };
                            let bi = ex.bound_ik[d];
                            let (d_o, d_i, d_p) = ((o - e).abs(), (iv - e).abs(), (o - iv).abs());
                            r.ours = r.ours.max(d_o / bo);
                            r.ik = r.ik.max(d_i / bi);
                            r.pair = r.pair.max(d_p / (bo + bi));
                            r.plain = r.plain.max(d_p / f64::from(mx_ik));
                            layer_ok &= d_o <= bo && d_i <= bi && d_p <= bo + bi;
                        }
                    }
                    if !layer_ok {
                        println!(
                            "flash set={label} layer={l} pass={} keys={live} ctx={ctx} rerun={rerun} \
                             nan_padding_same={nan_same} FAIL",
                            if mma { "mma" } else { "scalar" }
                        );
                    }
                    set_ok &= layer_ok;
                    ys.push(y);
                }

                if !graph_done {
                    for (pass, mma) in [false, true].into_iter().enumerate() {
                        let mut pv =
                            DeviceBuffer::<f32>::zeroed(stream, partials_v_len(n_head, ctx))?;
                        let mut pms =
                            DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(n_head, ctx))?;
                        let mut yg = DeviceBuffer::<f32>::zeroed(stream, n_head * HEAD)?;
                        let graph = gpu.capture(|s| {
                            k.enqueue_pass(
                                s,
                                GqaArgs {
                                    q: &inp.q,
                                    kc: &inp.kc,
                                    vc: &inp.vc,
                                    n_keys: &inp.n_keys,
                                    scale,
                                    n_kv,
                                    ctx,
                                    part_v: &mut pv,
                                    part_ms: &mut pms,
                                    y: &mut yg,
                                },
                                mma,
                            )
                        })?;
                        graph.launch(stream)?;
                        stream.synchronize()?;
                        let same = bits_equal(&yg.to_host_vec(stream)?, &ys[pass]);
                        let nodes = graph.node_count();
                        let pass_ok = same && nodes == 2;
                        println!(
                            "graph op={}+gqa_flash_merge set={label} layer={l} segments={} \
                             eager_vs_graph_bit_identical={same} graph_nodes={nodes} {}",
                            if mma {
                                "gqa_flash_seg_mma"
                            } else {
                                "gqa_flash_seg"
                            },
                            segments_for(ctx),
                            verdict(pass_ok)
                        );
                        ok &= pass_ok;
                    }
                    graph_done = true;
                }
                n_live = live;
                ctx_seen = ctx;
                layers += 1;
            }
            if layers == 0 {
                return Err(format!("{label}: no fa-0 row").into());
            }
            for (pass, r) in rs.iter().enumerate() {
                println!(
                    "flash set={label} pass={} layers={layers} keys={n_live} ctx={ctx_seen} segments={} — \
                     measured / bound: ours-exact {:.3e}, ik-exact {:.3e}, ours-ik {:.3e}; ours-ik plain \
                     {:.3e} of max|ik| (printed)",
                    if pass == 1 { "mma" } else { "scalar" },
                    segments_for(ctx_seen),
                    r.ours,
                    r.ik,
                    r.pair,
                    r.plain
                );
            }
            println!(
                "flash set={label}: bounds, reruns, NaN padding {}",
                verdict(set_ok)
            );
            ok &= set_ok;
        }
        println!("gate_qwen3moe_flash: {}", verdict(ok));
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }
}
