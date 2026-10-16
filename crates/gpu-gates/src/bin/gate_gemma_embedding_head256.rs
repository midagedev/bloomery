//! Head-256 prefill with three query heads per key head, against the f64
//! attention and first-order band in `flash_band`, over whole inputs and
//! symmetric windows. Block, slice and key-tile tails run on two key heads too.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_gemma_embedding_head256", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::Gpu;
    use bloomery_gpu::fault::{Fault, FaultSite, LAYER_NONE};
    use bloomery_gpu::flash_gqa_prefill::{FlashGqaPrefill, GqaPrefillArgs, HEAD_256, KEY_TILE};
    use bloomery_gpu_gates::flash_band::judge_prefill;
    use bloomery_gpu_gates::prefill_gate::{HostCache, cache};
    use bloomery_gpu_gates::{GateError, activations, bits_equal, checks_failed, verdict};
    use cuda_core::DeviceBuffer;

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let gpu = Gpu::new()?;
        let kernel = FlashGqaPrefill::load(gpu.context())?;
        let stream = gpu.stream();
        println!(
            "gate_gemma_embedding_head256: device {}",
            gpu.device_name()?
        );
        let mut ok = true;
        for n_kv in [1, 2] {
            let n_head = 3 * n_kv;
            let width = n_head * HEAD_256;
            for t in [1usize, 63, 65, 129] {
                let ctx = if t == 65 { t } else { t + KEY_TILE };
                let c = cache(stream, HEAD_256, n_kv, ctx, t, 700)?;
                let q = activations(HEAD_256, t * n_head, 703);
                let qd = DeviceBuffer::from_host(stream, &q)?;
                for reach in [t, 17, 64].into_iter().filter(|&r| r == t || r < t - 1) {
                    let first: Vec<u32> = (0..t)
                        .map(|i| u32::try_from(i.saturating_sub(reach)))
                        .collect::<Result<_, _>>()?;
                    let counts: Vec<u32> = (0..t)
                        .map(|i| u32::try_from((i + reach + 1).min(t)))
                        .collect::<Result<_, _>>()?;
                    let fd = DeviceBuffer::from_host(stream, &first)?;
                    let nd = DeviceBuffer::from_host(stream, &counts)?;
                    for scale in [1.0f32, 0.0625] {
                        let launch =
                            |kc: &DeviceBuffer<u16>,
                             vc: &DeviceBuffer<u16>,
                             nd: &DeviceBuffer<u32>,
                             fd: &DeviceBuffer<u32>,
                             out: &mut DeviceBuffer<f32>| {
                                kernel.enqueue_256_p1_lower(
                                    stream,
                                    GqaPrefillArgs {
                                        q: &qd,
                                        kc,
                                        vc,
                                        n_keys: nd,
                                        scale,
                                        n_head,
                                        n_kv,
                                        ctx,
                                        t,
                                        fault: gpu.unlabelled_sink(),
                                        y: out,
                                    },
                                    fd,
                                )
                            };
                        let mut yd = DeviceBuffer::from_host(stream, &vec![f32::NAN; t * width])?;
                        launch(&c.kc, &c.vc, &nd, &fd, &mut yd)?;
                        let y = yd.to_host_vec(stream)?;
                        launch(&c.kn, &c.vn, &nd, &fd, &mut yd)?;
                        let padded = yd.to_host_vec(stream)?;
                        launch(&c.kc, &c.vc, &nd, &fd, &mut yd)?;
                        let again = yd.to_host_vec(stream)?;
                        let mut band = true;
                        let mut worst = 0.0f64;
                        for r in 0..t {
                            let (lo, hi) = (first[r] as usize, counts[r] as usize);
                            let cut = |p: &[f32]| {
                                p.chunks(ctx * HEAD_256)
                                    .flat_map(|h| h[lo * HEAD_256..hi * HEAD_256].iter().copied())
                                    .collect()
                            };
                            let host = HostCache {
                                kf: cut(&c.host.kf),
                                vf: cut(&c.host.vf),
                                ctx: hi - lo,
                            };
                            let (pass, ratio) = judge_prefill::<HEAD_256, KEY_TILE>(
                                &q[r * width..(r + 1) * width],
                                &[hi - lo],
                                &host,
                                &y[r * width..(r + 1) * width],
                                scale,
                                (n_head, n_kv),
                                lo,
                            );
                            band &= pass;
                            worst = worst.max(ratio);
                        }
                        let graph = gpu.capture(|_| launch(&c.kc, &c.vc, &nd, &fd, &mut yd))?;
                        yd.copy_from_host(stream, &vec![f32::NAN; t * width])?;
                        graph.launch(stream)?;
                        let captured = yd.to_host_vec(stream)?;
                        let pass = band
                            && bits_equal(&y, &padded)
                            && bits_equal(&y, &again)
                            && bits_equal(&y, &captured)
                            && graph.node_count() == 1
                            && gpu.fault()?.is_none();
                        println!(
                            "prefill heads={n_head}/{n_kv} T={t} ctx={ctx} reach={reach} scale={scale}: measured/bound={worst:.3e} band={band} rerun={} nan_padding={} graph={} {}",
                            bits_equal(&y, &again),
                            bits_equal(&y, &padded),
                            bits_equal(&y, &captured),
                            verdict(pass)
                        );
                        ok &= pass;

                        if t == 65 && reach == 17 && scale == 1.0 {
                            let want = Some(Fault::at(LAYER_NONE, FaultSite::KeyCount));
                            for (row, lo, hi) in [
                                (3, counts[3], counts[3]),
                                (37, counts[37] + 1, counts[37]),
                                (3, first[3], 0),
                                (37, first[37], u32::try_from(ctx + 1)?),
                            ] {
                                let mut bad_first = first.clone();
                                let mut bad_counts = counts.clone();
                                bad_first[row] = lo;
                                bad_counts[row] = hi;
                                let bd = DeviceBuffer::from_host(stream, &bad_first)?;
                                let bn = DeviceBuffer::from_host(stream, &bad_counts)?;
                                yd.copy_from_host(stream, &y)?;
                                launch(&c.kc, &c.vc, &bn, &bd, &mut yd)?;
                                let out = yd.to_host_vec(stream)?;
                                let fault = gpu.take_fault()?;
                                let isolated =
                                    out.chunks(width).zip(y.chunks(width)).enumerate().all(
                                        |(r, (a, b))| {
                                            if r == row {
                                                a.iter().all(|v| v.is_nan())
                                            } else {
                                                bits_equal(a, b)
                                            }
                                        },
                                    );
                                let pass = fault == want && isolated;
                                println!(
                                    "prefill refusal row={row} first={lo} count={hi}: {}",
                                    verdict(pass)
                                );
                                ok &= pass;
                            }
                            let short = DeviceBuffer::from_host(stream, &first[..t - 1])?;
                            let refused = launch(&c.kc, &c.vc, &nd, &short, &mut yd);
                            let named = matches!(refused, Err(bloomery_gpu::GpuError::Shape { ref detail, .. }) if detail.starts_with("k_first.len()"));
                            println!("prefill short first-key buffer: {}", verdict(named));
                            ok &= named;
                        }
                    }
                }
            }
        }
        println!("gate_gemma_embedding_head256: {}", verdict(ok));
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
