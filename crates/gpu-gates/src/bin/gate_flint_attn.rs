//! Flint's head-128 attention kernels, model-free: the window-and-sinks flash
//! (`gqa_flash_seg_h128_swa`, `gqa_flash_merge_h128_swa`) and the no-rope q/k
//! RMS append (`qk_rms_append`).
//!
//! The flash is held to an f64 softmax of the same f16 keys and values, over
//! the gate's own window — the last `window` live keys, or every live key
//! when `window` is 0 or the count is shorter — plus the sink as one more
//! logit with value 0. The cut is not `window_cut`: a mutant of the kernel's
//! cut must not move the reference. The absolute band is 1e-2.
//! A key planted immediately before a window of 513 (count 600) and a run
//! with the sink removed are each far from that reference, so a window one
//! key too wide and a dropped sink both leave the band and land on the
//! reference they were supposed to miss.
//!
//! The RMS append is bit-identical to the kernel's own reduction (f32
//! squares, an f64 xor butterfly, warp 0 then warp 1, `(sum / 128) as f32`,
//! the positive-finite scale) with no rotation. A host rotation of that
//! result is far from the kernel's output. A position at `ctx` raises
//! `cache_pos`, NaNs the token and appends nothing.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_flint_attn", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::attention_h128::Kernels as LoadedAttention;
    use bloomery_gpu::fault::{Fault, FaultSite};
    use bloomery_gpu::flash_gqa::{GROUP, HEAD, window_segments};
    use bloomery_gpu::flash_gqa_h128_swa::{H128SwaArgs, H128SwaKernels};
    use bloomery_gpu::qk_rms::{HEAD as RMS_HEAD, QkRmsArgs, QkRmsKernels};
    use bloomery_gpu::rope_neox::NeoxArgs;
    use bloomery_gpu::{Gpu, GpuError, LAYER_NONE};
    use bloomery_gpu_gates::rounding::butterfly;
    use bloomery_gpu_gates::{GateError, bits_equal, checks_failed, verdict};
    use cuda_core::DeviceBuffer;
    use gguf::quant::{f32_to_f16_bits, half_to_f32};

    /// Absolute band of the flash against the f64 reference.
    const BAND: f32 = 1e-2;
    /// A mutant reference this far from the kernel is a different attention.
    const FAR: f32 = 5e-2;
    /// Live keys of the window clause, and the window those keys are cut by.
    const LONG: usize = 600;
    const WINDOW: usize = 513;
    /// The key immediately before `[LONG - WINDOW, LONG)`.
    const PLANT: usize = 86;
    const SINK: f32 = 8.0;
    const EPS: f32 = 1e-6;
    const SENTINEL: u16 = 0x5a5a;

    /// The gate's window, not the kernel's: the last `window` keys, or every
    /// key when there is no window or the row is shorter than one.
    fn window_first(count: usize, window: usize) -> usize {
        if window == 0 || count < window {
            0
        } else {
            count - window
        }
    }

    fn score_scale() -> f32 {
        1.0 / (HEAD as f32).sqrt()
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::INFINITY;
        }
        let mut m = 0.0f32;
        for (x, y) in a.iter().zip(b) {
            let d = (f64::from(*x) - f64::from(*y)).abs() as f32;
            if !d.is_finite() {
                return f32::INFINITY;
            }
            m = m.max(d);
        }
        m
    }

    /// f64 attention over keys `[start, end)` plus an optional sink. The sink
    /// contributes its logit and a zero value.
    fn attend(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        start: usize,
        end: usize,
        scale: f32,
        sink: Option<f32>,
    ) -> Vec<f32> {
        let mut logits = Vec::with_capacity(end - start + 1);
        for j in start..end {
            let mut dot = 0.0f64;
            for d in 0..HEAD {
                dot += f64::from(q[d]) * f64::from(half_to_f32(kc[j * HEAD + d]));
            }
            logits.push((f64::from(scale) * dot, Some(j)));
        }
        if let Some(s) = sink {
            logits.push((f64::from(s), None));
        }
        let mx = logits.iter().fold(f64::NEG_INFINITY, |m, (s, _)| m.max(*s));
        let mut denom = 0.0f64;
        let mut acc = vec![0.0f64; HEAD];
        for (logit, key) in &logits {
            let w = (*logit - mx).exp();
            denom += w;
            if let Some(j) = key {
                for d in 0..HEAD {
                    acc[d] += w * f64::from(half_to_f32(vc[j * HEAD + d]));
                }
            }
        }
        acc.iter().map(|a| (a / denom) as f32).collect()
    }

    fn fill_const(plane: &mut [u16], pos: usize, value: f32) {
        let bits = f32_to_f16_bits(value);
        plane[pos * HEAD..(pos + 1) * HEAD].fill(bits);
    }

    /// Inside keys are a small score; key [`PLANT`] is large, so attending it
    /// moves the output off the sink-dominated row.
    fn planes(ctx: usize) -> (Vec<u16>, Vec<u16>) {
        let mut kc = vec![0u16; ctx * HEAD];
        let mut vc = vec![0u16; ctx * HEAD];
        for pos in 0..ctx {
            if pos == PLANT {
                fill_const(&mut kc, pos, 4.0);
                fill_const(&mut vc, pos, 3.0);
            } else {
                fill_const(&mut kc, pos, 0.05);
                fill_const(&mut vc, pos, 0.5);
            }
        }
        (kc, vc)
    }

    fn clause(name: &str, pass: bool, detail: &str) -> bool {
        println!("{name} {detail} {}", verdict(pass));
        pass
    }

    fn refused(label: &str, r: Result<(), GpuError>, phrase: &str) -> bool {
        match r {
            Err(e) => {
                let msg = e.to_string();
                clause(
                    &format!("refusal {label}"),
                    msg.contains(phrase),
                    &format!("phrase {phrase:?} err {msg}"),
                )
            }
            Ok(()) => clause(
                &format!("refusal {label}"),
                false,
                &format!("expected an error containing {phrase:?}"),
            ),
        }
    }

    struct FlashIn<'a> {
        q: &'a DeviceBuffer<f32>,
        kc: &'a DeviceBuffer<u16>,
        vc: &'a DeviceBuffer<u16>,
        n_keys: &'a DeviceBuffer<u32>,
        sinks: Option<&'a DeviceBuffer<f32>>,
        scale: f32,
        n_kv: usize,
        n_head: usize,
        ctx: usize,
        m: usize,
        window: usize,
    }

    /// A launch the host refuses before it reads a buffer length. `m`, the
    /// group, the scale and the sinks are the fields those checks name.
    fn early<'a>(
        q: &'a DeviceBuffer<f32>,
        half: &'a DeviceBuffer<u16>,
        n_keys: &'a DeviceBuffer<u32>,
        sinks: Option<&'a DeviceBuffer<f32>>,
        n_head: usize,
        m: usize,
        scale: f32,
    ) -> FlashIn<'a> {
        FlashIn {
            q,
            kc: half,
            vc: half,
            n_keys,
            sinks,
            scale,
            n_kv: 1,
            n_head,
            ctx: 1,
            m,
            window: WINDOW,
        }
    }

    fn launch(
        gpu: &Gpu,
        k: &H128SwaKernels,
        inn: FlashIn<'_>,
        part_v: &mut DeviceBuffer<f32>,
        part_ms: &mut DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        k.enqueue(
            gpu.stream(),
            H128SwaArgs {
                q: inn.q,
                kc: inn.kc,
                vc: inn.vc,
                n_keys: inn.n_keys,
                scale: inn.scale,
                n_kv: inn.n_kv,
                ctx: inn.ctx,
                m: inn.m,
                window: inn.window,
                sinks: inn.sinks,
                part_v,
                part_ms,
                fault: gpu.unlabelled_sink(),
                y,
            },
            inn.n_head,
        )
    }

    fn flash_case(
        gpu: &Gpu,
        k: &H128SwaKernels,
        kc: &[u16],
        vc: &[u16],
        count: usize,
        window: usize,
    ) -> Result<bool, GateError> {
        let s = gpu.stream();
        let n_kv = 1usize;
        let n_head = GROUP;
        let ctx = LONG;
        let m = 1usize;
        let segs = window_segments(window);
        let q_h = vec![0.25f32; n_head * HEAD];
        let sink_h = vec![SINK; n_head];
        let q = DeviceBuffer::from_host(s, &q_h)?;
        let kc_d = DeviceBuffer::from_host(s, kc)?;
        let vc_d = DeviceBuffer::from_host(s, vc)?;
        let n_keys = DeviceBuffer::from_host(s, &[count as u32])?;
        let sinks = DeviceBuffer::from_host(s, &sink_h)?;
        let mut part_v = DeviceBuffer::<f32>::zeroed(s, m * n_head * segs * HEAD)?;
        let mut part_ms = DeviceBuffer::<f32>::zeroed(s, m * n_head * segs * 2)?;
        let mut y = DeviceBuffer::<f32>::zeroed(s, m * n_head * HEAD)?;
        gpu.take_fault()?;
        launch(
            gpu,
            k,
            FlashIn {
                q: &q,
                kc: &kc_d,
                vc: &vc_d,
                n_keys: &n_keys,
                sinks: Some(&sinks),
                scale: score_scale(),
                n_kv,
                n_head,
                ctx,
                m,
                window,
            },
            &mut part_v,
            &mut part_ms,
            &mut y,
        )?;
        s.synchronize()?;
        let got = y.to_host_vec(s)?;
        let fault = gpu.take_fault()?;
        let first = window_first(count, window);
        let scale = score_scale();
        let finite = got.iter().all(|v| v.is_finite());
        let mut worst = 0.0f32;
        let mut before = f32::INFINITY;
        let mut dropped = f32::INFINITY;
        for h in 0..n_head {
            let qh = &q_h[h * HEAD..(h + 1) * HEAD];
            let yh = &got[h * HEAD..(h + 1) * HEAD];
            let good = attend(qh, kc, vc, first, count, scale, Some(SINK));
            worst = worst.max(max_abs(yh, &good));
            if first > 0 {
                let wide = attend(qh, kc, vc, first - 1, count, scale, Some(SINK));
                before = before.min(max_abs(yh, &wide));
            }
            let bare = attend(qh, kc, vc, first, count, scale, None);
            dropped = dropped.min(max_abs(yh, &bare));
        }
        let mut ok = clause(
            &format!("flash window={window} count={count} close"),
            finite && fault.is_none() && worst <= BAND,
            &format!("max|d|={worst:.3e} band={BAND:.3e} finite={finite} fault={fault:?}"),
        );
        if first > 0 {
            ok &= clause(
                &format!("flash window={window} count={count} key before the window"),
                finite && before > FAR,
                &format!("min|d| vs include-before={before:.3e} far>{FAR:.3e}"),
            );
        }
        ok &= clause(
            &format!("flash window={window} count={count} dropped sink"),
            finite && dropped > FAR,
            &format!("min|d| vs no sink={dropped:.3e} far>{FAR:.3e}"),
        );
        Ok(ok)
    }

    fn flash_fault(
        gpu: &Gpu,
        k: &H128SwaKernels,
        count: u32,
        ctx: usize,
        label: &str,
    ) -> Result<bool, GateError> {
        let s = gpu.stream();
        let n_head = GROUP;
        let segs = window_segments(WINDOW);
        let q = DeviceBuffer::from_host(s, &vec![0.25f32; n_head * HEAD])?;
        let kc = DeviceBuffer::from_host(s, &vec![0u16; ctx * HEAD])?;
        let vc = DeviceBuffer::from_host(s, &vec![0u16; ctx * HEAD])?;
        let n_keys = DeviceBuffer::from_host(s, &[count])?;
        let sinks = DeviceBuffer::from_host(s, &vec![SINK; n_head])?;
        let mut part_v = DeviceBuffer::<f32>::zeroed(s, n_head * segs * HEAD)?;
        let mut part_ms = DeviceBuffer::<f32>::zeroed(s, n_head * segs * 2)?;
        let mut y = DeviceBuffer::<f32>::zeroed(s, n_head * HEAD)?;
        gpu.take_fault()?;
        launch(
            gpu,
            k,
            FlashIn {
                q: &q,
                kc: &kc,
                vc: &vc,
                n_keys: &n_keys,
                sinks: Some(&sinks),
                scale: score_scale(),
                n_kv: 1,
                n_head,
                ctx,
                m: 1,
                window: WINDOW,
            },
            &mut part_v,
            &mut part_ms,
            &mut y,
        )?;
        s.synchronize()?;
        let got = y.to_host_vec(s)?;
        let fault = gpu.take_fault()?;
        let want = Some(Fault::at(LAYER_NONE, FaultSite::KeyCount));
        Ok(clause(
            label,
            got.iter().all(|v| v.is_nan()) && fault == want,
            &format!(
                "nan_row={} fault={fault:?} want={want:?}",
                got.iter().all(|v| v.is_nan())
            ),
        ))
    }

    fn flash_refusals(gpu: &Gpu, k: &H128SwaKernels) -> Result<bool, GateError> {
        let s = gpu.stream();
        let one_f = DeviceBuffer::from_host(s, &[0.0f32])?;
        let one_h = DeviceBuffer::from_host(s, &[0u16])?;
        let one_u = DeviceBuffer::from_host(s, &[1u32])?;
        let mut part_v = DeviceBuffer::<f32>::zeroed(s, 1)?;
        let mut part_ms = DeviceBuffer::<f32>::zeroed(s, 1)?;
        let mut y = DeviceBuffer::<f32>::zeroed(s, 1)?;
        // The four checks below return before any buffer length is read.
        let mut ok = refused(
            "m = 0",
            launch(
                gpu,
                k,
                early(&one_f, &one_h, &one_u, Some(&one_f), GROUP, 0, 1.0),
                &mut part_v,
                &mut part_ms,
                &mut y,
            ),
            "m >= 1",
        );
        ok &= refused(
            "scale not finite",
            launch(
                gpu,
                k,
                early(&one_f, &one_h, &one_u, Some(&one_f), GROUP, 1, f32::NAN),
                &mut part_v,
                &mut part_ms,
                &mut y,
            ),
            "must be finite",
        );
        ok &= refused(
            "group not 8",
            launch(
                gpu,
                k,
                early(&one_f, &one_h, &one_u, Some(&one_f), 4, 1, 1.0),
                &mut part_v,
                &mut part_ms,
                &mut y,
            ),
            "exactly 8",
        );
        ok &= refused(
            "sinks missing",
            launch(
                gpu,
                k,
                early(&one_f, &one_h, &one_u, None, GROUP, 1, 1.0),
                &mut part_v,
                &mut part_ms,
                &mut y,
            ),
            "pass the layer's sinks",
        );

        let n_head = GROUP;
        let ctx = 8usize;
        let segs = window_segments(WINDOW);
        let q = DeviceBuffer::from_host(s, &[0.25f32])?;
        let kc = DeviceBuffer::from_host(s, &vec![0u16; ctx * HEAD])?;
        let vc = DeviceBuffer::from_host(s, &vec![0u16; ctx * HEAD])?;
        let n_keys = DeviceBuffer::from_host(s, &[1u32])?;
        let sinks = DeviceBuffer::from_host(s, &vec![SINK; n_head])?;
        let mut part_v = DeviceBuffer::<f32>::zeroed(s, n_head * segs * HEAD)?;
        let mut part_ms = DeviceBuffer::<f32>::zeroed(s, n_head * segs * 2)?;
        let mut y = DeviceBuffer::<f32>::zeroed(s, n_head * HEAD)?;
        ok &= refused(
            "q short",
            launch(
                gpu,
                k,
                FlashIn {
                    q: &q,
                    kc: &kc,
                    vc: &vc,
                    n_keys: &n_keys,
                    sinks: Some(&sinks),
                    scale: score_scale(),
                    n_kv: 1,
                    n_head,
                    ctx,
                    m: 1,
                    window: WINDOW,
                },
                &mut part_v,
                &mut part_ms,
                &mut y,
            ),
            "q.len()",
        );
        Ok(ok)
    }

    /// The kernel's reduction: per-thread f32 squares summed in f64, the xor
    /// butterfly inside each warp, warp 0 then warp 1, mean narrowed, scale
    /// `1/sqrt(mean+eps)` when that is finite and positive.
    fn rms_head(x: &[f32], gain: &[f32], eps: f32) -> Vec<f32> {
        let pair = |tid: usize| {
            let (x0, x1) = (x[tid], x[tid + RMS_HEAD / 2]);
            f64::from(x0 * x0) + f64::from(x1 * x1)
        };
        let w0 = butterfly(std::array::from_fn(&pair));
        let w1 = butterfly(std::array::from_fn(|lane| pair(32 + lane)));
        let mean = ((w0 + w1) / RMS_HEAD as f64) as f32;
        let scale = {
            let s = 1.0f32 / (mean + eps).sqrt();
            if s > 0.0 && s < f32::INFINITY {
                s
            } else {
                f32::NAN
            }
        };
        (0..RMS_HEAD).map(|d| (scale * gain[d]) * x[d]).collect()
    }

    fn rotated(n: &[f32]) -> Vec<f32> {
        const C: f32 = 0.6;
        const S: f32 = 0.8;
        let mut y = n.to_vec();
        for d in 0..RMS_HEAD / 2 {
            let (x0, x1) = (n[d], n[d + RMS_HEAD / 2]);
            y[d] = x0 * C - x1 * S;
            y[d + RMS_HEAD / 2] = x0 * S + x1 * C;
        }
        y
    }

    fn ramp(token: usize, head: usize, salt: f32) -> Vec<f32> {
        (0..RMS_HEAD)
            .map(|d| salt + token as f32 * 0.01 + head as f32 * 0.02 + (d as f32 + 1.0) * 0.001)
            .collect()
    }

    fn rms_case(gpu: &Gpu, k: &QkRmsKernels) -> Result<bool, GateError> {
        let s = gpu.stream();
        let (n_head, n_kv, ctx, m) = (8usize, 1usize, 8usize, 2usize);
        let good_pos = 4u32;
        let mut q_h = Vec::new();
        let mut k_h = Vec::new();
        let mut v_h = Vec::new();
        for t in 0..m {
            for h in 0..n_head {
                q_h.extend(ramp(t, h, 0.03));
            }
            k_h.extend(ramp(t, 0, 0.11));
            v_h.extend(ramp(t, 0, 0.17));
        }
        let gq_h: Vec<f32> = (0..RMS_HEAD).map(|d| 0.4 + d as f32 * 0.001).collect();
        let gk_h: Vec<f32> = (0..RMS_HEAD).map(|d| 0.7 + d as f32 * 0.002).collect();
        let pos_h = vec![good_pos, ctx as u32];
        let cache = vec![SENTINEL; n_kv * ctx * RMS_HEAD];
        let mut q = DeviceBuffer::from_host(s, &q_h)?;
        let mut kd = DeviceBuffer::from_host(s, &k_h)?;
        let v = DeviceBuffer::from_host(s, &v_h)?;
        let gq = DeviceBuffer::from_host(s, &gq_h)?;
        let gk = DeviceBuffer::from_host(s, &gk_h)?;
        let pos = DeviceBuffer::from_host(s, &pos_h)?;
        let mut cache_k = DeviceBuffer::from_host(s, &cache)?;
        let mut cache_v = DeviceBuffer::from_host(s, &cache)?;
        gpu.take_fault()?;
        k.enqueue(
            s,
            QkRmsArgs {
                q: &mut q,
                k: &mut kd,
                v: &v,
                gq: &gq,
                gk: &gk,
                pos: &pos,
                eps: EPS,
                n_head,
                n_kv,
                ctx,
                m,
                fault: gpu.unlabelled_sink(),
                cache_k: &mut cache_k,
                cache_v: &mut cache_v,
            },
        )?;
        s.synchronize()?;
        let (q_got, k_got) = (q.to_host_vec(s)?, kd.to_host_vec(s)?);
        let (ck, cv) = (cache_k.to_host_vec(s)?, cache_v.to_host_vec(s)?);
        let fault = gpu.take_fault()?;

        let mut q_same = true;
        let mut rot_gap = f32::INFINITY;
        for h in 0..n_head {
            let x = &q_h[h * RMS_HEAD..(h + 1) * RMS_HEAD];
            let got = &q_got[h * RMS_HEAD..(h + 1) * RMS_HEAD];
            let host = rms_head(x, &gq_h, EPS);
            q_same &= bits_equal(got, &host);
            rot_gap = rot_gap.min(max_abs(got, &rotated(&host)));
        }
        let k_host = rms_head(&k_h[..RMS_HEAD], &gk_h, EPS);
        let k_same = bits_equal(&k_got[..RMS_HEAD], &k_host);
        rot_gap = rot_gap.min(max_abs(&k_got[..RMS_HEAD], &rotated(&k_host)));
        let mut expect_k = cache.clone();
        let mut expect_v = cache;
        let row = good_pos as usize * RMS_HEAD;
        for d in 0..RMS_HEAD {
            expect_k[row + d] = f32_to_f16_bits(k_host[d]);
            expect_v[row + d] = f32_to_f16_bits(v_h[d]);
        }
        let cache_same = ck == expect_k && cv == expect_v;
        let bad_q = q_got[n_head * RMS_HEAD..].iter().all(|v| v.is_nan());
        let bad_k = k_got[RMS_HEAD..].iter().all(|v| v.is_nan());
        let want = Some(Fault::at(LAYER_NONE, FaultSite::CachePos));

        let mut ok = clause(
            "rms unrotated q/k and f16 cache",
            q_same && k_same && cache_same,
            &format!("q_heads={q_same} k={k_same} cache={cache_same}"),
        );
        ok &= clause(
            "rms far from a host rotation",
            q_same && k_same && rot_gap > FAR,
            &format!("min|d| vs c=0.6 s=0.8 is {rot_gap:.3e}"),
        );
        ok &= clause(
            "rms position at ctx",
            bad_q && bad_k && cache_same && fault == want,
            &format!(
                "q_nan={bad_q} k_nan={bad_k} cache_untouched_elsewhere={cache_same} fault={fault:?}"
            ),
        );

        let mut q = DeviceBuffer::from_host(s, &[0.0f32])?;
        let mut kd = DeviceBuffer::from_host(s, &[0.0f32])?;
        let v = DeviceBuffer::from_host(s, &[0.0f32])?;
        let gq = DeviceBuffer::from_host(s, &[0.0f32])?;
        let gk = DeviceBuffer::from_host(s, &[0.0f32])?;
        let pos = DeviceBuffer::from_host(s, &[0u32])?;
        let mut cache_k = DeviceBuffer::from_host(s, &[0u16])?;
        let mut cache_v = DeviceBuffer::from_host(s, &[0u16])?;
        ok &= refused(
            "rms m > ctx",
            k.enqueue(
                s,
                QkRmsArgs {
                    q: &mut q,
                    k: &mut kd,
                    v: &v,
                    gq: &gq,
                    gk: &gk,
                    pos: &pos,
                    eps: EPS,
                    n_head: 1,
                    n_kv: 1,
                    ctx: 1,
                    m: 2,
                    fault: gpu.unlabelled_sink(),
                    cache_k: &mut cache_k,
                    cache_v: &mut cache_v,
                },
            ),
            "1 <= m <= ctx",
        );
        Ok(ok)
    }

    fn sample(pos: usize, head: usize, dim: usize, salt: usize) -> f32 {
        ((pos * 17 + head * 13 + dim * 7 + salt) % 97) as f32 / 97.0 - 0.5
    }

    fn turn_at(values: &[f32], table: &[f32], pos: usize, rotate: bool) -> Vec<f32> {
        let mut out = values.to_vec();
        if rotate {
            for d in 0..HEAD / 2 {
                let (x, y) = (values[d], values[d + HEAD / 2]);
                let (c, s) = (table[pos * HEAD + 2 * d], table[pos * HEAD + 2 * d + 1]);
                out[d] = x.mul_add(c, -(y * s));
                out[d + HEAD / 2] = x.mul_add(s, y * c);
            }
        }
        out
    }

    /// Append across a wrap before a causal multirow pass. The full logical planes are the
    /// independent oracle; only the device ring maps absolute rows modulo its capacity.
    fn loaded_case(
        gpu: &Gpu,
        loaded: &LoadedAttention,
        rotate: bool,
        window: usize,
        rows: usize,
        m: usize,
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let ctx = 3072usize;
        let start = 2047usize;
        let kv = 2;
        let heads = kv * GROUP;
        let gq_h: Vec<f32> = (0..HEAD).map(|d| 0.6 + d as f32 * 0.001).collect();
        let gk_h: Vec<f32> = (0..HEAD).map(|d| 0.8 - d as f32 * 0.001).collect();
        let mut table_h = vec![0.0f32; ctx * HEAD];
        for p in 0..ctx {
            for d in 0..HEAD / 2 {
                let angle = p as f64 * 10000.0f64.powf(-((2 * d) as f64) / HEAD as f64);
                table_h[p * HEAD + 2 * d] = angle.cos() as f32;
                table_h[p * HEAD + 2 * d + 1] = angle.sin() as f32;
            }
        }
        let mut logical_k = vec![0u16; kv * ctx * HEAD];
        let mut logical_v = vec![0u16; kv * ctx * HEAD];
        for kh in 0..kv {
            for p in 0..start + m {
                let raw: Vec<f32> = (0..HEAD).map(|d| sample(p, kh, d, 11)).collect();
                let key = turn_at(&rms_head(&raw, &gk_h, EPS), &table_h, p, rotate);
                for d in 0..HEAD {
                    logical_k[(kh * ctx + p) * HEAD + d] = f32_to_f16_bits(key[d]);
                    logical_v[(kh * ctx + p) * HEAD + d] =
                        f32_to_f16_bits(sample(p, kh, d, 29) * 4.0);
                }
            }
        }
        let guard = 2 * HEAD;
        let mut cache_k_h = vec![SENTINEL; kv * rows * HEAD + guard];
        let mut cache_v_h = cache_k_h.clone();
        for kh in 0..kv {
            for p in start.saturating_sub(rows)..start {
                let physical = (kh * rows + p % rows) * HEAD;
                let logical = (kh * ctx + p) * HEAD;
                cache_k_h[physical..physical + HEAD]
                    .copy_from_slice(&logical_k[logical..logical + HEAD]);
                cache_v_h[physical..physical + HEAD]
                    .copy_from_slice(&logical_v[logical..logical + HEAD]);
            }
        }
        let mut expect_k = cache_k_h.clone();
        let mut expect_v = cache_v_h.clone();
        let mut q_h = Vec::new();
        let mut raw_k = Vec::new();
        let mut raw_v = Vec::new();
        let mut expect_q = Vec::new();
        for t in 0..m {
            let p = start + t;
            for h in 0..heads {
                let raw: Vec<f32> = (0..HEAD).map(|d| sample(p, h, d, 3)).collect();
                expect_q.extend(turn_at(&rms_head(&raw, &gq_h, EPS), &table_h, p, rotate));
                q_h.extend(raw);
            }
            for kh in 0..kv {
                raw_k.extend((0..HEAD).map(|d| sample(p, kh, d, 11)));
                raw_v.extend((0..HEAD).map(|d| sample(p, kh, d, 29) * 4.0));
                let physical = (kh * rows + p % rows) * HEAD;
                let logical = (kh * ctx + p) * HEAD;
                expect_k[physical..physical + HEAD]
                    .copy_from_slice(&logical_k[logical..logical + HEAD]);
                expect_v[physical..physical + HEAD]
                    .copy_from_slice(&logical_v[logical..logical + HEAD]);
            }
        }
        let pos_h: Vec<u32> = (start..start + m).map(|p| p as u32).collect();
        let counts_h: Vec<u32> = pos_h.iter().map(|p| p + 1).collect();
        let mut q = DeviceBuffer::from_host(stream, &q_h)?;
        let mut kd = DeviceBuffer::from_host(stream, &raw_k)?;
        let v = DeviceBuffer::from_host(stream, &raw_v)?;
        let gq = DeviceBuffer::from_host(stream, &gq_h)?;
        let gk = DeviceBuffer::from_host(stream, &gk_h)?;
        let table = DeviceBuffer::from_host(stream, &table_h)?;
        let pos = DeviceBuffer::from_host(stream, &pos_h)?;
        let counts = DeviceBuffer::from_host(stream, &counts_h)?;
        let sinks_h: Vec<f32> = (0..heads).map(|h| 7.0 + h as f32 * 0.3).collect();
        let sinks = DeviceBuffer::from_host(stream, &sinks_h)?;
        let mut ck = DeviceBuffer::from_host(stream, &cache_k_h)?;
        let mut cv = DeviceBuffer::from_host(stream, &cache_v_h)?;
        let segs = window_segments(window);
        let mut pv = DeviceBuffer::zeroed(stream, m * heads * segs * HEAD)?;
        let mut pm = DeviceBuffer::zeroed(stream, m * heads * segs * 2)?;
        let mut out = DeviceBuffer::from_host(stream, &vec![12345.0f32; m * heads * HEAD + HEAD])?;
        gpu.take_fault()?;
        // SAFETY: consecutive positions, distinct modulo rows, and m <= rows-window+1
        // for rings. All readable historical keys above were initialized.
        unsafe {
            loaded.enqueue_append(
                stream,
                NeoxArgs {
                    q: &mut q,
                    k: &mut kd,
                    v: &v,
                    gq: &gq,
                    gk: &gk,
                    table: &table,
                    pos: &pos,
                    eps: EPS,
                    n_head: heads,
                    n_kv: kv,
                    ctx,
                    m,
                    fault: gpu.unlabelled_sink(),
                    cache_k: &mut ck,
                    cache_v: &mut cv,
                },
                rows,
                rotate,
            )?;
        }
        loaded.enqueue_attention(
            stream,
            H128SwaArgs {
                q: &q,
                kc: &ck,
                vc: &cv,
                n_keys: &counts,
                scale: score_scale(),
                n_kv: kv,
                ctx,
                m,
                window,
                sinks: Some(&sinks),
                part_v: &mut pv,
                part_ms: &mut pm,
                fault: gpu.unlabelled_sink(),
                y: &mut out,
            },
            heads,
            rows,
        )?;
        stream.synchronize()?;
        let got_q = q.to_host_vec(stream)?;
        let got = out.to_host_vec(stream)?;
        let cache_same = ck.to_host_vec(stream)? == expect_k && cv.to_host_vec(stream)? == expect_v;
        let query_same = bits_equal(&got_q, &expect_q);
        let mut error = 0.0f32;
        let mut no_sink_gap = 0.0f32;
        for t in 0..m {
            let end = start + t + 1;
            for (h, &sink) in sinks_h.iter().enumerate() {
                let b = (t * heads + h) * HEAD;
                let base = (h / GROUP) * ctx * HEAD;
                let keys = &logical_k[base..base + ctx * HEAD];
                let values = &logical_v[base..base + ctx * HEAD];
                let reference = attend(
                    &expect_q[b..b + HEAD],
                    keys,
                    values,
                    window_first(end, window),
                    end,
                    score_scale(),
                    Some(sink),
                );
                error = error.max(max_abs(&got[b..b + HEAD], &reference));
                if t == 0 && h == 0 {
                    let no_sink = attend(
                        &expect_q[b..b + HEAD],
                        keys,
                        values,
                        window_first(end, window),
                        end,
                        score_scale(),
                        None,
                    );
                    no_sink_gap = max_abs(&reference, &no_sink);
                }
            }
        }
        let canary = got[m * heads * HEAD..].iter().all(|x| *x == 12345.0);
        let clean = gpu.take_fault()?.is_none();
        let mut ok = clause(
            "loaded attention logical/ring",
            query_same && cache_same && canary && clean && error <= BAND,
            &format!(
                "rotate={rotate} window={window} rows={rows} m={m} q={query_same} cache={cache_same} canary={canary} clean={clean} max_abs={error:.3e}"
            ),
        );
        ok &= clause(
            "loaded attention sink witness",
            no_sink_gap > 1e-5,
            &format!("max_abs without sink={no_sink_gap:.3e}"),
        );
        if rows < ctx {
            ok &= refused(
                "ring unsafe append-then-attend chunk",
                loaded.enqueue_attention(
                    stream,
                    H128SwaArgs {
                        q: &q,
                        kc: &ck,
                        vc: &cv,
                        n_keys: &counts,
                        scale: score_scale(),
                        n_kv: kv,
                        ctx,
                        m: rows - window + 2,
                        window,
                        sinks: Some(&sinks),
                        part_v: &mut pv,
                        part_ms: &mut pm,
                        fault: gpu.unlabelled_sink(),
                        y: &mut out,
                    },
                    heads,
                    rows,
                ),
                "invalid ring geometry",
            );
            let bad_counts = DeviceBuffer::from_host(stream, &[0u32, ctx as u32 + 1])?;
            loaded.enqueue_attention(
                stream,
                H128SwaArgs {
                    q: &q,
                    kc: &ck,
                    vc: &cv,
                    n_keys: &bad_counts,
                    scale: score_scale(),
                    n_kv: kv,
                    ctx,
                    m: 2,
                    window,
                    sinks: Some(&sinks),
                    part_v: &mut pv,
                    part_ms: &mut pm,
                    fault: gpu.unlabelled_sink(),
                    y: &mut out,
                },
                heads,
                rows,
            )?;
            stream.synchronize()?;
            let invalid = out.to_host_vec(stream)?;
            ok &= clause(
                "ring invalid logical count",
                invalid[..2 * heads * HEAD].iter().all(|v| v.is_nan())
                    && gpu.take_fault()? == Some(Fault::at(LAYER_NONE, FaultSite::KeyCount)),
                "zero and count past logical ctx are NaN with key_count",
            );
            let invalid_pos = DeviceBuffer::from_host(stream, &[ctx as u32])?;
            let before_k = ck.to_host_vec(stream)?;
            let before_v = cv.to_host_vec(stream)?;
            // SAFETY: the out-of-range position is never appended.
            unsafe {
                loaded.enqueue_append(
                    stream,
                    NeoxArgs {
                        q: &mut q,
                        k: &mut kd,
                        v: &v,
                        gq: &gq,
                        gk: &gk,
                        table: &table,
                        pos: &invalid_pos,
                        eps: EPS,
                        n_head: heads,
                        n_kv: kv,
                        ctx,
                        m: 1,
                        fault: gpu.unlabelled_sink(),
                        cache_k: &mut ck,
                        cache_v: &mut cv,
                    },
                    rows,
                    rotate,
                )?;
            }
            stream.synchronize()?;
            ok &= clause(
                "ring invalid absolute position",
                q.to_host_vec(stream)?[..heads * HEAD]
                    .iter()
                    .all(|v| v.is_nan())
                    && ck.to_host_vec(stream)? == before_k
                    && cv.to_host_vec(stream)? == before_v
                    && gpu.take_fault()? == Some(Fault::at(LAYER_NONE, FaultSite::CachePos)),
                "position at logical ctx is NaN and preserves all physical rows",
            );
        }
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let gpu = Gpu::new()?;
        println!("gate_flint_attn: device {}", gpu.device_name()?);
        let flash = H128SwaKernels::load(gpu.context())?;
        let rms = QkRmsKernels::load(gpu.context())?;
        let cut_ok = window_first(LONG, WINDOW) == PLANT + 1
            && window_first(40, WINDOW) == 0
            && window_first(40, 0) == 0;
        let mut ok = clause(
            "flash gate cut",
            cut_ok,
            "600/513 starts at 87, a short row and a full layer start at 0",
        );
        let (kc, vc) = planes(LONG);
        ok &= flash_case(&gpu, &flash, &kc, &vc, LONG, WINDOW)?;
        ok &= flash_case(&gpu, &flash, &kc, &vc, 40, WINDOW)?;
        ok &= flash_case(&gpu, &flash, &kc, &vc, 40, 0)?;
        ok &= flash_fault(&gpu, &flash, 0, 8, "flash count 0")?;
        ok &= flash_fault(&gpu, &flash, 9, 8, "flash count past ctx")?;
        ok &= flash_refusals(&gpu, &flash)?;
        ok &= rms_case(&gpu, &rms)?;
        let loaded = LoadedAttention::load(&gpu)?;
        for rotate in [false, true] {
            ok &= loaded_case(&gpu, &loaded, rotate, WINDOW, 1024, 3)?;
            ok &= loaded_case(&gpu, &loaded, rotate, WINDOW, 1024, 512)?;
            ok &= loaded_case(&gpu, &loaded, rotate, 0, 3072, 3)?;
            ok &= loaded_case(&gpu, &loaded, rotate, WINDOW, 3072, 3)?;
        }
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
