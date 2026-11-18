//! GPU gate for DeepSeek-V4.1's router and experts — B4 op blocks J and K
//! (`docs/research/v41-b4-plan-report.md` §1-J, §1-K) — on the file's own
//! weights with seeded input, every layer, each token launched alone (the
//! decode shape of the kernels).
//!
//! PIN(2026-11-18): the real-file ik set retired; ik cannot read bq3km's engram schema; no fixture set carries this clause
//!
//! The input, per layer: `TOKENS` seeded activations of the model's width
//! stand for the norm's output (`activations`), the router kernel's own ids
//! on them pick each token's experts, and seeded activations of the experts'
//! width stand for the routed gate·up output and the shared expert's. What a
//! kernel is held to here is its arithmetic and its structure, which any
//! finite input of the right shape carries.
//!
//! Sites, per layer:
//! - `router`: `ds41_router` — `ffn_moe_logits`, `_probs`, `_topk` and
//!   `_weights_scaled`, every layer;
//! - `routed`: `ds41_expert_gate_up` for `ffn_moe_gate_par` and `q4k_sel`'s
//!   gemv for `ffn_moe_down`, at the layers whose down stack is q4_K — a
//!   q5_K down (layers 0 and 1) runs the fused dispatch the lib gate pins,
//!   not this site's launches;
//! - `shared`: the shared expert's gate·up·SwiGLU for `ffn_up_gate` and its
//!   down projection for `ffn_shexp`, every layer, in the file's format —
//!   `ds41_shexp_gate_up` and `q8_0_gemv` on a q8_0 file,
//!   `ds41_shexp_gate_up_q3k` on the q8_1 form of the input and the
//!   Q4_K gemv on the q8_1 form of h (`ds41_q5k_gemv_f32` on h for a Q5_K
//!   down) on a K-quant one;
//! - `cardrule`: qdot's card-rule host kernels (q8_1 per 128 values, the
//!   Q3_K/Q4_K lane walks and the warp tree) against the card's own bytes at
//!   every q4_K-down layer — the x and h quantizer planes, the fused
//!   gate·up's h, the sel gemv's down, bit for bit — and once a run the
//!   grouped tile path (`ds41_expert_gate_up_tiles`, `q4k_gemv_tiles`) at
//!   two columns an expert against the host's one-column calls;
//!
//! PIN(2026-09-24): removed — the `combine` site and the combine launch of the layer graph pinned `ds41_moe_combine`, which the engine never runs (its combine is `ds41_ffn_post`'s, pinned by the MoE chain gate against `ffn_out`); the kernel is gone.
//!
//! The routed and shared lines count where the clamp bites on the op path's
//! dots (`clamp_hits`: `silu(g) > L` / `u > L` / `u < -L`).
//!
//! One comparison per site: **the kernel against the host transcription of
//! our rule** — bit-identical where the host runs the rule itself: the SwiGLU
//! on the op-path kernels' dots (`q3k_gemv_sel`, `q3k_gemv`, `q8_0_gemv`: the
//! same bodies and warp tree), the selection and the weights from the
//! kernel's own scores; each dot within `KERNEL_BAND` of its f64 value (rows
//! decoded as `dequant_row` does, against the activation our rule reads: q8_1
//! per 128 values for q3_K and q4_K, f32 for q8_0, q5_K and the router); the
//! router's logits bit-identical to `f32_gemv`'s; the scores within the
//! distance of the device's `expf`/`logf` from the host's (`softplus_err`);
//! every kernel rerun bit-identical.
//!
//! Synthetic checks: router ties planted where the selection treats them
//! differently (one lane, adjacent lanes, the two ends, three ways, the
//! 6th/7th boundary, all equal, the bias deciding, every score 0), against
//! the rule's statement, the weights finite; weight rows NaN in all but four
//! experts, where six slots cannot be filled and the router must raise its
//! fault (`FaultSite::Router`) rather than fill the last two with expert 0;
//! the clamp crossed on `silu(g)`,
//! on `u > L` and on `u < -L`, and `L = 0` (no clamp), routed and shared,
//! against the host and against the clamp's statement in f64
//! (`stated_ratio`); an out-of-range expert id; one layer's seven launches
//! (eight where the shared down reads q8_1) captured as a graph and
//! replayed.

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_deepseek41_moe", gate::run())
}

#[cfg(feature = "deepseek41")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "deepseek41")]
mod gate {
    use bloomery_gpu::kquant::{TileTable, TiledDown};
    use bloomery_gpu::q4k_sel::tile_cap;
    use bloomery_gpu::route_core::renorm_divisor;
    use bloomery_gpu::weights::{DevWeight, Q8Block, q8_0_planes};
    use bloomery_gpu::{DeviceTensor, Fault, FaultSite, Gpu, LAYER_NONE, Q8Act};
    use bloomery_gpu_deepseek41::chain::ffn::{FfnBatchKernels, TiledGateUp};
    use bloomery_gpu_deepseek41::dense::{Dense, DenseKernels};
    use bloomery_gpu_deepseek41::experts::{ExpertGateUp, ExpertKernels, silu_ik, swiglu_clamp};
    use bloomery_gpu_deepseek41::router::{
        N_EXPERT, N_USED, RouterKernels, RouterOut, sqrt_softplus,
    };
    use bloomery_gpu_gates::act_rule::Rule;
    use bloomery_gpu_gates::rounding::U;
    use bloomery_gpu_gates::{
        GateError, KERNEL_BAND, activations, bits_equal, bytes_to_words, checks_failed,
        max_rel_err, open_split, q8_1_dequant, row_bytes, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::quant::{GgmlType, dequant_row};
    use gguf::{Split, Value};
    use model::arch::Arch;

    /// One side's f32 SwiGLU against the exact function at its own inputs,
    /// relative: `v_expf` is within 1.45358 + 0.5 ulp (ik's own comment on
    /// it), so `e^-g` is within 3.92u of itself; `1 + e` and the division
    /// round once each (silu within 5.92u); the clamps are exact and `· u`
    /// rounds once: 6.92u, below 7u.
    const SILU_ROUNDING: f64 = 7.0 * U;

    /// Half the spacing of f32's subnormals, 2^-150: a product that lands
    /// below `f32::MIN_POSITIVE` rounds by at most this much, whatever its
    /// relative error.
    const HALF_SUBNORMAL: f64 = f32::from_bits(1) as f64 / 2.0;

    /// CUDA's `expf` and `logf` (the libdevice bodies the kernel calls): at
    /// most 2 and 1 ulp (CUDA C++ Programming Guide, mathematical functions).
    /// The kernel's `1 + expf(x)` is fused into `expf`'s last step and rounds
    /// once, from the unrounded exponential — 2.5 ulp from `e^x`.
    const DEVICE_EXP_ULPS: f64 = 2.5;
    const DEVICE_LOG_ULPS: f64 = 1.0;

    /// The host's `expf`/`logf` (glibc, which ik's CPU build calls too): at
    /// most 1 ulp each, glibc's documented bound.
    const HOST_EXP_ULPS: f64 = 1.0;
    const HOST_LOG_ULPS: f64 = 1.0;

    /// The file's MoE constants.
    struct Meta {
        n_layer: usize,
        n_embd: usize,
        n_ff: usize,
        scale: f32,
        gating: u64,
        /// `swiglu_clamp_exp` and `swiglu_clamp_shexp`, one per layer.
        clamp_exp: Vec<f32>,
        clamp_shexp: Vec<f32>,
    }

    impl Meta {
        fn read(split: &Split) -> Result<Meta, GateError> {
            let u = |s: &str| -> Result<u64, GateError> {
                split
                    .arch_get_u64(s)
                    .ok_or_else(|| format!("metadata {} missing", split.arch_key(s)).into())
            };
            let (n_expert, n_used) = (u("expert_count")?, u("expert_used_count")?);
            if n_expert != N_EXPERT as u64 || n_used != N_USED as u64 {
                return Err(format!(
                    "expert_count {n_expert}, expert_used_count {n_used}: the router \
                     kernel is built for {N_EXPERT} and {N_USED}"
                )
                .into());
            }
            if u("expert_shared_count")? != 1 {
                return Err("expert_shared_count is not 1".into());
            }
            let norm_key = split.arch_key("expert_weights_norm");
            if split.value(&norm_key).and_then(Value::as_bool) != Some(true) {
                return Err(format!("{norm_key} is not true: the rule renormalizes").into());
            }
            let scale = split.arch_get_f32("expert_weights_scale").ok_or_else(|| {
                format!(
                    "metadata {} missing",
                    split.arch_key("expert_weights_scale")
                )
            })?;
            let n_layer = usize::try_from(u("block_count")?)?;
            let clamp = |s: &str| -> Result<Vec<f32>, GateError> {
                let key = split.arch_key(s);
                let Some(Value::Array(v)) = split.value(&key) else {
                    return Err(format!("metadata {key} missing or not an array").into());
                };
                let v: Option<Vec<f32>> = v.iter().map(Value::as_f32).collect();
                match v {
                    Some(v) if v.len() == n_layer => Ok(v),
                    _ => Err(format!("{key}: want {n_layer} f32 values").into()),
                }
            };
            Ok(Meta {
                n_layer,
                n_embd: usize::try_from(u("embedding_length")?)?,
                n_ff: usize::try_from(u("expert_feed_forward_length")?)?,
                scale,
                gating: u("expert_gating_func")?,
                clamp_exp: clamp("swiglu_clamp_exp")?,
                clamp_shexp: clamp("swiglu_clamp_shexp")?,
            })
        }
    }

    /// A tensor's file bytes, its type and dims checked.
    fn tensor<'a>(
        split: &'a Split,
        name: &str,
        ty: GgmlType,
        dims: &[u64],
    ) -> Result<&'a [u8], GateError> {
        let (shard, info) = split
            .find(name)
            .ok_or_else(|| format!("tensor {name} is not in the model"))?;
        if info.ty != ty || info.dims != dims {
            return Err(format!(
                "tensor {name} is {} {:?}, want {ty} {dims:?}",
                info.ty, info.dims
            )
            .into());
        }
        let g = split
            .shard(shard)
            .ok_or_else(|| format!("tensor {name}: shard {shard} missing"))?;
        Ok(g.data(info)?)
    }

    /// `f(i, chunk)` over the `len`-element chunks of `out`, on every core.
    fn par_chunks<T: Send>(
        out: &mut [T],
        len: usize,
        f: impl Fn(usize, &mut [T]) -> Result<(), String> + Sync,
    ) -> Result<(), GateError> {
        let n = out.len() / len.max(1);
        if n == 0 {
            return Ok(());
        }
        let threads = std::thread::available_parallelism()
            .map_or(1, |t| t.get())
            .min(n);
        let per = n.div_ceil(threads);
        let errs: Vec<String> = std::thread::scope(|s| {
            let handles: Vec<_> = out
                .chunks_mut(per * len)
                .enumerate()
                .map(|(c, block)| {
                    let f = &f;
                    s.spawn(move || -> Result<(), String> {
                        for (j, chunk) in block.chunks_mut(len).enumerate() {
                            f(c * per + j, chunk)?;
                        }
                        Ok(())
                    })
                })
                .collect();
            handles
                .into_iter()
                .filter_map(|h| match h.join() {
                    Ok(Ok(())) => None,
                    Ok(Err(e)) => Some(e),
                    Err(_) => Some("a worker thread panicked".to_string()),
                })
                .collect()
        });
        match errs.into_iter().next() {
            None => Ok(()),
            Some(e) => Err(e.into()),
        }
    }

    /// Rows `r0 .. r0 + n` of a tensor of `k`-value rows of type `ty`, decoded
    /// to f32 by `dequant_row`.
    fn rows_f32(
        ty: GgmlType,
        bytes: &[u8],
        r0: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, GateError> {
        let rb = row_bytes(ty, k)?;
        let src = bytes
            .get(r0 * rb..(r0 + n) * rb)
            .ok_or_else(|| format!("rows {r0}..{} of {ty} x {k}: past the tensor", r0 + n))?;
        let mut out = vec![0.0f32; n * k];
        par_chunks(&mut out, k, |r, dst| {
            dequant_row(ty, &src[r * rb..(r + 1) * rb], dst).map_err(|e| e.to_string())
        })?;
        Ok(out)
    }

    /// A q8_0 tensor of `rows` rows of `k` values as the q8f32 two-plane
    /// device weight.
    fn q8_0_weight(
        stream: &CudaStream,
        bytes: &[u8],
        rows: usize,
        k: usize,
    ) -> Result<DevWeight, GateError> {
        let blocks: Vec<Q8Block> = bytes
            .as_chunks::<34>()
            .0
            .iter()
            .map(Q8Block::from_bytes)
            .collect();
        if blocks.len() != rows * k / 32 {
            return Err(format!(
                "q8_0 tensor holds {} blocks, want {}",
                blocks.len(),
                rows * k / 32
            )
            .into());
        }
        let (qs, d) = q8_0_planes(&blocks);
        Ok(DevWeight::Q8_0 {
            qs: DeviceTensor::upload(stream, &qs, rows, k / 4)?,
            d: DeviceTensor::upload(stream, &d, rows, k / 32)?,
            k,
        })
    }

    /// Every row of `w` (rows of `x.len()` values) dotted with `x` in f64, each
    /// narrowed to f32: the value a kernel's dot over the same activation is
    /// held to. The products of two f32 are exact in f64; the f64 sums round
    /// `k` times at 2^-53, far below `KERNEL_BAND`.
    fn row_dots(w: &[f32], x: &[f32]) -> Result<Vec<f32>, GateError> {
        let k = x.len();
        let mut out = vec![0.0f32; w.len() / k];
        par_chunks(&mut out, 1, |r, o| {
            let s: f64 = w[r * k..(r + 1) * k]
                .iter()
                .zip(x)
                .map(|(&wv, &a)| f64::from(wv) * f64::from(a))
                .sum();
            o[0] = s as f32;
            Ok(())
        })?;
        Ok(out)
    }

    /// The largest `|a - b| / band` (0 where both are 0, infinite where only
    /// the band is).
    fn worst(ratio: f64, diff: f64, band: f64) -> f64 {
        let r = if diff == 0.0 {
            0.0
        } else if band > 0.0 {
            diff / band
        } else {
            f64::INFINITY
        };
        if r.is_nan() {
            f64::INFINITY
        } else {
            ratio.max(r)
        }
    }

    fn silu64(g: f64) -> f64 {
        g / (1.0 + (-g).exp())
    }

    /// The exact clamped SwiGLU.
    fn swiglu64(g: f64, u: f64, limit: f32) -> f64 {
        let (mut s, mut uc) = (silu64(g), u);
        if limit > 1e-6 {
            let l = f64::from(limit);
            s = s.min(l);
            uc = uc.clamp(-l, l);
        }
        s * uc
    }

    /// One f32 ulp at `v`: the spacing of f32 values at `|v|`'s binade (the
    /// subnormal spacing below the normal range).
    fn ulp32(v: f64) -> f64 {
        let a = v.abs() as f32;
        if a < f32::MIN_POSITIVE {
            return f64::from(f32::from_bits(1));
        }
        f64::from(f32::from_bits(a.to_bits() & 0x7f80_0000)) * f64::from(f32::EPSILON)
    }

    /// The exact `sqrt(ln(1 + e^x))`, in f64.
    fn softplus64(x: f32) -> f64 {
        let x = f64::from(x);
        if x > 20.0 {
            x.sqrt()
        } else {
            (1.0 + x.exp()).ln().sqrt()
        }
    }

    /// How far one side's f32 score can sit from the exact score at `x`,
    /// given its `expf` and `logf` ulp bounds: `e^x` off by `e_ulps` ulp and
    /// `1 + e` rounded once (half an ulp of `y`); `ln` has slope `1/y` there
    /// and adds `l_ulps` ulp; `sqrt` moves a change `δ` of its argument by at
    /// most `δ / sqrt(sp)` and rounds once. Above 20 the score is `sqrt(x)`,
    /// one rounding.
    fn softplus_err(x: f32, e_ulps: f64, l_ulps: f64) -> f64 {
        let p = softplus64(x);
        if x > 20.0 {
            return 0.5 * ulp32(p);
        }
        let e = f64::from(x).exp();
        let y = 1.0 + e;
        let sp = y.ln();
        let dy = e_ulps * ulp32(e) + 0.5 * ulp32(y);
        let dsp = dy / (y - dy) + l_ulps * ulp32(sp);
        dsp / p + 0.5 * ulp32(p)
    }

    /// The first `N_USED` indices of `key` ordered by the key descending, an
    /// equal key to the larger index — `std::greater` on `(value, index)`
    /// pairs. `+ 0.0` folds -0.0 into +0.0, so `total_cmp` ties exactly the
    /// keys `==` ties (the callers hold no NaN).
    fn top_by<T: Copy + std::ops::Add<Output = T> + From<f32>>(
        key: &[T],
        cmp: impl Fn(&T, &T) -> std::cmp::Ordering,
    ) -> [u32; N_USED] {
        let k = |i: u32| key[i as usize] + T::from(0.0);
        let mut idx: Vec<u32> = (0..key.len() as u32).collect();
        idx.sort_by(|&a, &b| cmp(&k(b), &k(a)).then(b.cmp(&a)));
        let mut out = [0u32; N_USED];
        out.copy_from_slice(&idx[..N_USED]);
        out
    }

    /// ik's selection, stated: [`top_by`] over the f32 selection values.
    fn select(v: &[f32]) -> Result<[u32; N_USED], GateError> {
        if let Some(i) = v.iter().position(|x| !x.is_finite()) {
            return Err(format!("select: value {i} is not finite").into());
        }
        Ok(top_by(v, f32::total_cmp))
    }

    /// ik's weights from the scores `p` at `ids`: gathered, summed in f64 in
    /// slot order and narrowed, each divided by the sum's [`renorm_divisor`],
    /// then scaled. Returns (scaled, sum, divided).
    fn weights_rule(p: &[f32], ids: &[u32], scale: f32) -> ([f32; N_USED], f32, [f32; N_USED]) {
        let mut g = [0.0f32; N_USED];
        for (o, &i) in g.iter_mut().zip(ids) {
            *o = p[i as usize];
        }
        let mut sum = 0.0f64;
        for &v in &g {
            sum += f64::from(v);
        }
        let sum = sum as f32;
        let div = renorm_divisor(sum);
        let norm = g.map(|v| v / div);
        (norm.map(|v| v * scale), sum, norm)
    }

    /// Device scratch, allocated once.
    struct Dev {
        x: DeviceBuffer<f32>,
        rout: RouterOut,
        logits_op: DeviceBuffer<f32>,
        act_x: Q8Act,
        g6: DeviceBuffer<f32>,
        u6: DeviceBuffer<f32>,
        h6: DeviceBuffer<f32>,
        hin: DeviceBuffer<f32>,
        act_h: Q8Act,
        d6: DeviceBuffer<f32>,
        sg: DeviceBuffer<f32>,
        su: DeviceBuffer<f32>,
        sh: DeviceBuffer<f32>,
        shin: DeviceBuffer<f32>,
        /// The shared down's input in q8_1, where its weight reads one.
        act_sh: Q8Act,
        sy: DeviceBuffer<f32>,
    }

    impl Dev {
        fn new(stream: &CudaStream, m: &Meta) -> Result<Dev, GateError> {
            let z = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
            Ok(Dev {
                x: z(m.n_embd)?,
                rout: RouterOut::new(stream)?,
                logits_op: z(N_EXPERT)?,
                act_x: Q8Act::with_k(stream, 1, m.n_embd)?,
                g6: z(N_USED * m.n_ff)?,
                u6: z(N_USED * m.n_ff)?,
                h6: z(N_USED * m.n_ff)?,
                hin: z(N_USED * m.n_ff)?,
                act_h: Q8Act::with_k(stream, N_USED, m.n_ff)?,
                d6: z(N_USED * m.n_embd)?,
                sg: z(m.n_ff)?,
                su: z(m.n_ff)?,
                sh: z(m.n_ff)?,
                shin: z(m.n_ff)?,
                act_sh: Q8Act::with_k(stream, 1, m.n_ff)?,
                sy: z(m.n_embd)?,
            })
        }
    }

    /// What every site reads: the model, the device and the kernels.
    struct Cx {
        split: Split,
        meta: Meta,
        gpu: Gpu,
        router: RouterKernels,
        experts: ExpertKernels,
        dense: DenseKernels,
    }

    /// One layer's router and shared-expert weights: f32 host copies for the
    /// f64 references and the device copies the kernels read.
    struct Layer {
        l: usize,
        router_w: Vec<f32>,
        router_dev: DeviceTensor<f32>,
        bias: Vec<f32>,
        bias_dev: DeviceBuffer<f32>,
        sh_gate: ShW,
        sh_up: ShW,
        sh_down: ShW,
        /// The down stack's type: q4_K puts the routed experts on the card.
        down_ty: GgmlType,
    }

    impl Layer {
        fn load(cx: &Cx, l: usize) -> Result<Layer, GateError> {
            let (k, ff) = (cx.meta.n_embd, cx.meta.n_ff);
            let (k64, ne64) = (k as u64, N_EXPERT as u64);
            let stream = cx.gpu.stream();
            let rb = tensor(
                &cx.split,
                &format!("blk.{l}.ffn_gate_inp.weight"),
                GgmlType::BF16,
                &[k64, ne64],
            )?;
            let router_w = rows_f32(GgmlType::BF16, rb, 0, N_EXPERT, k)?;
            let bb = tensor(
                &cx.split,
                &format!("blk.{l}.exp_probs_b.bias"),
                GgmlType::F32,
                &[ne64],
            )?;
            let bias = rows_f32(GgmlType::F32, bb, 0, 1, N_EXPERT)?;
            let q = |name: &str, rows: usize, kk: usize, formats: &[GgmlType]| {
                ShW::load(cx, &format!("blk.{l}.{name}.weight"), rows, kk, formats)
            };
            let gate_up = [GgmlType::Q8_0, GgmlType::Q3_K];
            let sh_gate = q("ffn_gate_shexp", ff, k, &gate_up)?;
            let sh_up = q("ffn_up_shexp", ff, k, &gate_up)?;
            if sh_up.ty != sh_gate.ty {
                return Err(format!(
                    "layer {l}: ffn_gate_shexp is {} and ffn_up_shexp {}: the fused gate·up \
                     reads one format",
                    sh_gate.ty, sh_up.ty
                )
                .into());
            }
            let sh_down = q(
                "ffn_down_shexp",
                k,
                ff,
                &[GgmlType::Q8_0, GgmlType::Q4_K, GgmlType::Q5_K],
            )?;
            let down_name = format!("blk.{l}.ffn_down_exps.weight");
            let down_ty = cx
                .split
                .find(&down_name)
                .map(|(_, t)| t.ty)
                .ok_or_else(|| format!("tensor {down_name} is not in the model"))?;
            Ok(Layer {
                l,
                router_dev: DeviceTensor::upload(stream, &router_w, N_EXPERT, k)?,
                router_w,
                bias_dev: DeviceBuffer::from_host(stream, &bias)?,
                bias,
                sh_gate,
                sh_up,
                sh_down,
                down_ty,
            })
        }
    }

    /// A shared-expert weight: its format, its rows decoded to f32 for the
    /// f64 references and its device copy in the format its kernel loads.
    struct ShW {
        ty: GgmlType,
        rows: Vec<f32>,
        dev: DevWeight,
    }

    impl ShW {
        /// Weight `name`, `rows` rows of `kk` values, in one of `formats`;
        /// any other format is refused with the tensor named.
        fn load(
            cx: &Cx,
            name: &str,
            rows: usize,
            kk: usize,
            formats: &[GgmlType],
        ) -> Result<ShW, GateError> {
            let ty = cx
                .split
                .find(name)
                .map(|(_, t)| t.ty)
                .ok_or_else(|| format!("tensor {name} is not in the model"))?;
            if !formats.contains(&ty) {
                return Err(format!(
                    "{name} is {ty}, want one of {formats:?}: the gate has no rule for it"
                )
                .into());
            }
            let b = tensor(&cx.split, name, ty, &[kk as u64, rows as u64])?;
            let stream = cx.gpu.stream();
            let dev = if ty == GgmlType::Q8_0 {
                q8_0_weight(stream, b, rows, kk)?
            } else {
                if !b.len().is_multiple_of(4 * rows) {
                    return Err(
                        format!("{name}: {} bytes are not {rows} word rows", b.len()).into(),
                    );
                }
                let words = bytes_to_words(b);
                let w = DeviceTensor::upload(stream, &words, rows, words.len() / rows)?;
                DevWeight::KQuant { ty, w, k: kk }
            };
            Ok(ShW {
                ty,
                rows: rows_f32(ty, b, 0, rows, kk)?,
                dev,
            })
        }

        /// The weight as `dense`'s projection, for the down's kernel.
        fn dense(&self) -> Result<Dense<'_>, bloomery_gpu::GpuError> {
            match (&self.dev, self.ty) {
                (DevWeight::Q8_0 { qs, d, .. }, _) => Ok(Dense::Q8_0 { qs, d }),
                (DevWeight::KQuant { w, .. }, GgmlType::Q3_K) => Ok(Dense::Q3K(w)),
                (DevWeight::KQuant { w, .. }, GgmlType::Q4_K) => Ok(Dense::Q4K(w)),
                (DevWeight::KQuant { w, .. }, GgmlType::Q5_K) => Ok(Dense::Q5K(w)),
                _ => Err(bloomery_gpu::GpuError::Shape {
                    what: "ShW::dense",
                    detail: format!(
                        "a shared-expert weight of type {} has no dense kernel",
                        self.ty
                    ),
                }),
            }
        }

        /// The q8_0 planes, for the op path's q8_0 gemv.
        fn planes(
            &self,
        ) -> Result<(&DeviceTensor<u32>, &DeviceTensor<u16>), bloomery_gpu::GpuError> {
            match self.dense()? {
                Dense::Q8_0 { qs, d } => Ok((qs, d)),
                _ => Err(bloomery_gpu::GpuError::Shape {
                    what: "ShW::planes",
                    detail: format!("a shared-expert weight is {}, not q8_0", self.ty),
                }),
            }
        }

        /// The Q3_K word rows, for the op path's and the fused gate·up's
        /// Q3_K kernels.
        fn q3k(&self) -> Result<&DeviceTensor<u32>, bloomery_gpu::GpuError> {
            match self.dense()? {
                Dense::Q3K(w) => Ok(w),
                _ => Err(bloomery_gpu::GpuError::Shape {
                    what: "ShW::q3k",
                    detail: format!("a shared-expert weight is {}, not Q3_K", self.ty),
                }),
            }
        }
    }

    /// The activation our rule reads under a shared-expert weight of type
    /// `ty`, as the values its codes stand for: f32 under a q8_0 weight, the
    /// q8_1 form under a K-quant one ([`Rule`]).
    fn shared_act(ty: GgmlType, x: &[f32]) -> Vec<f32> {
        match Rule::of(ty) {
            Some(r) => r.ours.reconstruct(x),
            None => x.to_vec(),
        }
    }

    /// Enqueue the shared gate·up·SwiGLU of `x` in its weights' format: the
    /// q8_0 kernel on `x`, the Q3_K one on `act`, `x`'s q8_1 form (the
    /// caller quantized it).
    fn enqueue_shared_gate_up(
        cx: &Cx,
        ly: &Layer,
        x: &DeviceBuffer<f32>,
        act: &Q8Act,
        limit: f32,
        h: &mut DeviceBuffer<f32>,
    ) -> Result<(), bloomery_gpu::GpuError> {
        let stream = cx.gpu.stream();
        if ly.sh_gate.ty == GgmlType::Q3_K {
            let (g, u) = (ly.sh_gate.q3k()?, ly.sh_up.q3k()?);
            cx.dense.enqueue_shexp_gate_up_q3k(
                stream,
                g,
                u,
                act,
                limit,
                cx.gpu.unlabelled_sink(),
                h,
            )
        } else {
            cx.experts.enqueue_shexp_gate_up(
                stream,
                &ly.sh_gate.dev,
                &ly.sh_up.dev,
                x,
                limit,
                cx.gpu.unlabelled_sink(),
                h,
            )
        }
    }

    /// Enqueue the op path's shared gate and up dots of `x` (`act` its q8_1
    /// form, for Q3_K) into `g` and `u`: the gemvs whose bodies the fused
    /// kernel runs.
    fn enqueue_shared_dots(
        cx: &Cx,
        ly: &Layer,
        x: &DeviceBuffer<f32>,
        act: &Q8Act,
        g: &mut DeviceBuffer<f32>,
        u: &mut DeviceBuffer<f32>,
    ) -> Result<(), bloomery_gpu::GpuError> {
        if ly.sh_gate.ty == GgmlType::Q3_K {
            cx.gpu.enqueue_gemv_q3k(ly.sh_gate.q3k()?, act, g)?;
            cx.gpu.enqueue_gemv_q3k(ly.sh_up.q3k()?, act, u)
        } else {
            let (stream, q8) = (cx.gpu.stream(), cx.gpu.q8f32());
            let ((gq, gd), (uq, ud)) = (ly.sh_gate.planes()?, ly.sh_up.planes()?);
            q8.enqueue_q8_0_gemv(stream, gq, gd, x, 1, g)?;
            q8.enqueue_q8_0_gemv(stream, uq, ud, x, 1, u)
        }
    }

    /// Enqueue the shared down projection of `h` into `y` in its weight's
    /// format, `h`'s q8_1 form into `act` first where the weight reads one.
    fn enqueue_shared_down(
        cx: &Cx,
        ly: &Layer,
        h: &DeviceBuffer<f32>,
        act: &mut Q8Act,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), bloomery_gpu::GpuError> {
        let d = ly.sh_down.dense()?;
        let act = if d.reads_q8_1() {
            cx.gpu.enqueue_quantize_q8_1(h, act)?;
            Some(&*act)
        } else {
            None
        };
        cx.dense.enqueue(&cx.gpu, d, h, act, y)
    }

    #[derive(Default)]
    struct Tally {
        sites: u32,
        failed: u32,
    }

    impl Tally {
        fn add(&mut self, pass: bool) {
            self.sites += 1;
            if !pass {
                self.failed += 1;
            }
        }
    }

    pub fn run() -> Result<(), GateError> {
        let split = open_split(Arch::Deepseek41, "gate-gpu-ds41-moe")?;
        let meta = Meta::read(&split)?;
        let gpu = Gpu::new()?;
        let router = RouterKernels::load(gpu.context())?;
        let experts = ExpertKernels::load(gpu.context())?;
        let dense = DenseKernels::load(gpu.context())?;
        println!(
            "gate_deepseek41_moe: device {} — layers {} n_embd {} n_ff {} experts {N_EXPERT} used {N_USED} \
             gating_func {} scale {} clamp_exp[0] {} clamp_shexp[0] {}",
            gpu.device_name()?,
            meta.n_layer,
            meta.n_embd,
            meta.n_ff,
            meta.gating,
            meta.scale,
            meta.clamp_exp[0],
            meta.clamp_shexp[0]
        );
        let cx = Cx {
            split,
            meta,
            gpu,
            router,
            experts,
            dense,
        };
        let stream = cx.gpu.stream();
        let mut dv = Dev::new(stream, &cx.meta)?;

        let mut tally = Tally::default();
        tally.add(router_ties(&cx, &mut dv)?);
        tally.add(router_nan(&cx, &mut dv)?);

        let mut first_routed = true;
        for l in 0..cx.meta.n_layer {
            let ly = Layer::load(&cx, l)?;
            let input = seeded_input(&cx, &ly, &mut dv)?;
            tally.add(router_site(&cx, &ly, &input, &mut dv)?);
            tally.add(shared_site(&cx, &ly, &input, &mut dv)?);
            // PIN(2026-11-18): the real-file ik set retired; ik cannot read bq3km's engram schema; no fixture set carries this clause
            if ly.down_ty == GgmlType::Q4_K {
                for p in routed_layer(&cx, &ly, &input, &mut dv, first_routed)? {
                    tally.add(p);
                }
                first_routed = false;
            } else {
                println!(
                    "routed layer={l} skipped: ffn_down_exps is {} — this site pins the q4_K \
                     down's launches alone; a q5_K down runs the fused dispatch the lib gate pins",
                    ly.down_ty
                );
            }
        }
        if first_routed {
            return Err("no layer has a q4_K down stack: the routed kernels were not gated".into());
        }

        let pass = tally.failed == 0;
        println!(
            "gate_deepseek41_moe: {} sites across {} layers of {TOKENS} seeded tokens, {} failed — {}",
            tally.sites,
            cx.meta.n_layer,
            tally.failed,
            verdict(pass)
        );
        if !pass {
            return Err(checks_failed());
        }
        Ok(())
    }

    /// The router at one layer over the seeded tokens: the kernel against the
    /// host transcription of our rule.
    fn router_site(cx: &Cx, ly: &Layer, input: &Input, dv: &mut Dev) -> Result<bool, GateError> {
        let stream = cx.gpu.stream();
        let (k, l, scale) = (cx.meta.n_embd, ly.l, cx.meta.scale);
        let mut ok1 = true; // logits bits = f32_gemv, ids, weights
        let mut rel1 = 0.0f32; // logits vs f64
        let mut p1 = 0.0f64; // scores vs host, ratio
        let mut rerun = true;
        let mut tickets_zero = true;
        for xt in input.x.chunks_exact(k) {
            dv.x.copy_from_host(stream, xt)?;
            cx.router.enqueue_router(
                stream,
                &ly.router_dev,
                &dv.x,
                &ly.bias_dev,
                scale,
                &mut dv.rout,
                cx.gpu.unlabelled_sink(),
            )?;
            let lk = dv.rout.logits.to_host_vec(stream)?;
            let pk = dv.rout.probs.to_host_vec(stream)?;
            let idk = dv.rout.ids.to_host_vec(stream)?;
            let wk = dv.rout.weights.to_host_vec(stream)?;
            tickets_zero &= dv.rout.tickets(stream)? == 0;
            cx.router.enqueue_router(
                stream,
                &ly.router_dev,
                &dv.x,
                &ly.bias_dev,
                scale,
                &mut dv.rout,
                cx.gpu.unlabelled_sink(),
            )?;
            rerun &= bits_equal(&lk, &dv.rout.logits.to_host_vec(stream)?)
                && bits_equal(&pk, &dv.rout.probs.to_host_vec(stream)?)
                && idk == dv.rout.ids.to_host_vec(stream)?
                && bits_equal(&wk, &dv.rout.weights.to_host_vec(stream)?);
            tickets_zero &= dv.rout.tickets(stream)? == 0;
            cx.gpu
                .q8f32()
                .enqueue_f32_gemv(stream, &ly.router_dev, &dv.x, 1, &mut dv.logits_op)?;
            ok1 &= bits_equal(&lk, &dv.logits_op.to_host_vec(stream)?);

            let ours = row_dots(&ly.router_w, xt)?;
            rel1 = rel1.max(max_rel_err(&lk, &ours)?);
            for (&xl, &pv) in lk.iter().zip(&pk) {
                let band = softplus_err(xl, DEVICE_EXP_ULPS, DEVICE_LOG_ULPS)
                    + softplus_err(xl, HOST_EXP_ULPS, HOST_LOG_ULPS);
                p1 = worst(
                    p1,
                    (f64::from(pv) - f64::from(sqrt_softplus(xl))).abs(),
                    band,
                );
            }
            let biased_k: Vec<f32> = pk.iter().zip(&ly.bias).map(|(&p, &b)| p + b).collect();
            let sel_k = select(&biased_k)?;
            ok1 &= idk == sel_k;
            let (wk_host, _, _) = weights_rule(&pk, &idk, scale);
            ok1 &= bits_equal(&wk, &wk_host);
            // PIN(2026-11-18): the real-file ik set retired; ik cannot read bq3km's engram schema; no fixture set carries this clause
        }
        let pass = ok1 && rel1 <= KERNEL_BAND && p1 <= 1.0 && rerun && tickets_zero;
        println!(
            "router layer={l} tokens={TOKENS} logits_eq_f32_gemv+ids+weights_exact={ok1} logits_rel={rel1:.3e} \
             probs_ratio={p1:.3} rerun={rerun} tickets_zero={tickets_zero} {}",
            verdict(pass)
        );
        Ok(pass)
    }

    /// The shared expert at one layer over the seeded tokens: the kernels
    /// against the host transcription of our rule, the down on seeded h.
    fn shared_site(cx: &Cx, ly: &Layer, input: &Input, dv: &mut Dev) -> Result<bool, GateError> {
        let stream = cx.gpu.stream();
        let (k, ff, l) = (cx.meta.n_embd, cx.meta.n_ff, ly.l);
        let limit = cx.meta.clamp_shexp[l];
        let mut h_exact = true;
        let (mut g_rel, mut u_rel, mut y_rel) = (0.0f32, 0.0f32, 0.0f32);
        let mut rerun = true;
        let mut hits = [0usize; 3];
        for (xt, ht) in input.x.chunks_exact(k).zip(input.sh.chunks_exact(ff)) {
            dv.x.copy_from_host(stream, xt)?;
            if ly.sh_gate.ty == GgmlType::Q3_K {
                cx.gpu.enqueue_quantize_q8_1(&dv.x, &mut dv.act_x)?;
            }
            enqueue_shared_dots(cx, ly, &dv.x, &dv.act_x, &mut dv.sg, &mut dv.su)?;
            enqueue_shared_gate_up(cx, ly, &dv.x, &dv.act_x, limit, &mut dv.sh)?;
            let (g_op, u_op, hk) = (
                dv.sg.to_host_vec(stream)?,
                dv.su.to_host_vec(stream)?,
                dv.sh.to_host_vec(stream)?,
            );
            enqueue_shared_gate_up(cx, ly, &dv.x, &dv.act_x, limit, &mut dv.sh)?;
            rerun &= bits_equal(&hk, &dv.sh.to_host_vec(stream)?);
            dv.shin.copy_from_host(stream, ht)?;
            enqueue_shared_down(cx, ly, &dv.shin, &mut dv.act_sh, &mut dv.sy)?;
            let yk = dv.sy.to_host_vec(stream)?;
            enqueue_shared_down(cx, ly, &dv.shin, &mut dv.act_sh, &mut dv.sy)?;
            rerun &= bits_equal(&yk, &dv.sy.to_host_vec(stream)?);

            let h_host: Vec<f32> = g_op
                .iter()
                .zip(&u_op)
                .map(|(&g, &u)| swiglu_clamp(g, u, limit))
                .collect();
            h_exact &= bits_equal(&hk, &h_host);
            let (cs, cu, cl) = crossings(&g_op, &u_op, limit);
            hits = [hits[0] + cs, hits[1] + cu, hits[2] + cl];
            let xo = shared_act(ly.sh_gate.ty, xt);
            let g64 = row_dots(&ly.sh_gate.rows, &xo)?;
            let u64_ = row_dots(&ly.sh_up.rows, &xo)?;
            g_rel = g_rel.max(max_rel_err(&g_op, &g64)?);
            u_rel = u_rel.max(max_rel_err(&u_op, &u64_)?);
            let ho = shared_act(ly.sh_down.ty, ht);
            y_rel = y_rel.max(max_rel_err(&yk, &row_dots(&ly.sh_down.rows, &ho)?)?);
            // PIN(2026-11-18): the real-file ik set retired; ik cannot read bq3km's engram schema; no fixture set carries this clause
        }
        let pass = h_exact
            && g_rel <= KERNEL_BAND
            && u_rel <= KERNEL_BAND
            && y_rel <= KERNEL_BAND
            && rerun;
        println!(
            "shared layer={l} tokens={TOKENS} limit={limit} clamp_hits={}/{}/{} h_eq_host={h_exact} \
             gate_rel={g_rel:.3e} up_rel={u_rel:.3e} down_rel={y_rel:.3e} rerun={rerun} {}",
            hits[0],
            hits[1],
            hits[2],
            verdict(pass)
        );
        Ok(pass)
    }

    /// Seeded tokens a layer, each launched alone.
    const TOKENS: usize = 6;

    /// The activations' seeds (`activations`), each offset by the layer: the
    /// router's input, the routed h, the shared h.
    const SEED_X: u32 = 0xD541_0A01;
    const SEED_H: u32 = 0xD541_0B02;
    const SEED_SH: u32 = 0xD541_0C03;

    /// One layer's seeded input.
    struct Input {
        /// [`TOKENS`] activations of `n_embd`: the norm's output.
        x: Vec<f32>,
        /// The router kernel's ids on `x`, `N_USED` a token.
        ids: Vec<u32>,
        /// `TOKENS * N_USED` activations of `n_ff`: the routed gate·up output.
        h: Vec<f32>,
        /// `TOKENS` activations of `n_ff`: the shared gate·up output.
        sh: Vec<f32>,
    }

    /// Layer `ly`'s seeded input; each token's experts are the router
    /// kernel's own pick on its activation, every id proven in range.
    fn seeded_input(cx: &Cx, ly: &Layer, dv: &mut Dev) -> Result<Input, GateError> {
        let stream = cx.gpu.stream();
        let (k, ff, l) = (cx.meta.n_embd, cx.meta.n_ff, ly.l);
        let salt = u32::try_from(l)?;
        let x = activations(k, TOKENS, SEED_X.wrapping_add(salt));
        let mut ids = Vec::with_capacity(TOKENS * N_USED);
        for xt in x.chunks_exact(k) {
            dv.x.copy_from_host(stream, xt)?;
            cx.router.enqueue_router(
                stream,
                &ly.router_dev,
                &dv.x,
                &ly.bias_dev,
                cx.meta.scale,
                &mut dv.rout,
                cx.gpu.unlabelled_sink(),
            )?;
            ids.extend(dv.rout.ids.to_host_vec(stream)?);
        }
        if let Some(bad) = ids.iter().find(|&&id| id as usize >= N_EXPERT) {
            return Err(format!(
                "layer {l}: the router picked expert {bad}, past the file's {N_EXPERT}"
            )
            .into());
        }
        Ok(Input {
            x,
            ids,
            h: activations(ff, N_USED * TOKENS, SEED_H.wrapping_add(salt)),
            sh: activations(ff, TOKENS, SEED_SH.wrapping_add(salt)),
        })
    }

    /// A layer's three routed stacks in the file: gate and up q3_K, down q4_K.
    struct RoutedFile<'a> {
        gate: &'a [u8],
        up: &'a [u8],
        down: &'a [u8],
        /// Bytes of a gate/up row (`n_embd` values) and of a down row
        /// (`n_ff` values).
        rb_gu: usize,
        rb_d: usize,
    }

    impl<'a> RoutedFile<'a> {
        fn open(cx: &'a Cx, l: usize) -> Result<RoutedFile<'a>, GateError> {
            let (k, ff) = (cx.meta.n_embd, cx.meta.n_ff);
            let (k64, ff64, ne64) = (k as u64, ff as u64, N_EXPERT as u64);
            let gu = |name: &str| {
                tensor(
                    &cx.split,
                    &format!("blk.{l}.{name}.weight"),
                    GgmlType::Q3_K,
                    &[k64, ff64, ne64],
                )
            };
            let down_name = format!("blk.{l}.ffn_down_exps.weight");
            Ok(RoutedFile {
                gate: gu("ffn_gate_exps")?,
                up: gu("ffn_up_exps")?,
                down: tensor(&cx.split, &down_name, GgmlType::Q4_K, &[ff64, k64, ne64])?,
                rb_gu: row_bytes(GgmlType::Q3_K, k)?,
                rb_d: row_bytes(GgmlType::Q4_K, ff)?,
            })
        }

        /// The stacks of `experts` (ascending ids), compacted and uploaded.
        fn upload(&self, cx: &Cx, experts: Vec<u32>) -> Result<Stacks, GateError> {
            let (k, ff) = (cx.meta.n_embd, cx.meta.n_ff);
            let stream = cx.gpu.stream();
            let stack = |b: &[u8], rows: usize, rb: usize| -> Vec<u32> {
                let mut out = Vec::with_capacity(experts.len() * rows * rb);
                for &e in &experts {
                    let e = e as usize;
                    out.extend_from_slice(&b[e * rows * rb..(e + 1) * rows * rb]);
                }
                bytes_to_words(&out)
            };
            let n = experts.len();
            Ok(Stacks {
                wg: DeviceTensor::upload(
                    stream,
                    &stack(self.gate, ff, self.rb_gu),
                    n * ff,
                    self.rb_gu / 4,
                )?,
                wu: DeviceTensor::upload(
                    stream,
                    &stack(self.up, ff, self.rb_gu),
                    n * ff,
                    self.rb_gu / 4,
                )?,
                wd: DeviceTensor::upload(
                    stream,
                    &stack(self.down, k, self.rb_d),
                    n * k,
                    self.rb_d / 4,
                )?,
                experts,
            })
        }
    }

    /// A layer's routed stacks on the card: the experts the seeded tokens
    /// use, in id order.
    struct Stacks {
        wg: DeviceTensor<u32>,
        wu: DeviceTensor<u32>,
        wd: DeviceTensor<u32>,
        experts: Vec<u32>,
    }

    impl Stacks {
        /// Token `tt`'s slots as places in the compact stacks (`u32::MAX`
        /// for an expert not resident).
        fn sel(&self, ids: &[u32], tt: usize) -> Vec<u32> {
            ids[tt * N_USED..(tt + 1) * N_USED]
                .iter()
                .map(|id| {
                    self.experts
                        .binary_search(id)
                        .map_or(u32::MAX, |p| p as u32)
                })
                .collect()
        }
    }

    /// The f64 value of every row of each dot of one (token, slot), in f32:
    /// gate and up on the q8_1 form of the token's x, down on the q8_1 form of
    /// the slot's h.
    #[derive(Default, Clone)]
    struct SlotRefs {
        g: Vec<f32>,
        u: Vec<f32>,
        d: Vec<f32>,
    }

    /// Every (token, slot)'s reference rows, expert by expert — each expert's
    /// weights decoded once.
    fn routed_refs(
        cx: &Cx,
        file: &RoutedFile<'_>,
        input: &Input,
        experts: &[u32],
    ) -> Result<Vec<SlotRefs>, GateError> {
        let (k, ff) = (cx.meta.n_embd, cx.meta.n_ff);
        let xo = q8_1_dequant(&input.x, k, TOKENS);
        let mut refs = vec![SlotRefs::default(); input.ids.len()];
        for &e in experts {
            let eu = e as usize;
            let wg = rows_f32(GgmlType::Q3_K, file.gate, eu * ff, ff, k)?;
            let wu = rows_f32(GgmlType::Q3_K, file.up, eu * ff, ff, k)?;
            let wd = rows_f32(GgmlType::Q4_K, file.down, eu * k, k, ff)?;
            for (j, _) in input.ids.iter().enumerate().filter(|&(_, &id)| id == e) {
                let tt = j / N_USED;
                let xt = &xo[tt * k..(tt + 1) * k];
                let ho = q8_1_dequant(&input.h[j * ff..(j + 1) * ff], ff, 1);
                refs[j] = SlotRefs {
                    g: row_dots(&wg, xt)?,
                    u: row_dots(&wu, xt)?,
                    d: row_dots(&wd, &ho)?,
                };
            }
        }
        Ok(refs)
    }

    /// The routed experts at one layer: the stacks of the experts the seeded
    /// tokens use, compacted and uploaded once. Returns the sites' verdicts
    /// and, at the first routed layer, the synthetic checks'.
    fn routed_layer(
        cx: &Cx,
        ly: &Layer,
        input: &Input,
        dv: &mut Dev,
        synthetic: bool,
    ) -> Result<Vec<bool>, GateError> {
        let l = ly.l;
        let mut experts = input.ids.clone();
        experts.sort_unstable();
        experts.dedup();
        let file = RoutedFile::open(cx, l)?;
        let refs = routed_refs(cx, &file, input, &experts)?;
        let stacks = file.upload(cx, experts)?;
        let mut verdicts = vec![
            routed_set(cx, ly, input, &refs, &stacks, dv)?,
            cardrule_set(cx, ly, input, &stacks, &file, dv)?,
        ];
        if synthetic {
            let k = cx.meta.n_embd;
            let (x, sel) = (&input.x[..k], stacks.sel(&input.ids, 0));
            verdicts.push(clamp_routed(cx, dv, &stacks, &sel, x, l)?);
            verdicts.push(clamp_shared(cx, dv, ly, x)?);
            verdicts.push(out_of_range(
                cx,
                dv,
                &stacks,
                &sel,
                x,
                cx.meta.clamp_exp[l],
            )?);
            verdicts.push(graph(cx, dv, ly, &stacks, &sel, x)?);
            verdicts.push(cardrule_ties(cx, dv)?);
            verdicts.push(cardrule_tiles(cx, ly, &stacks, &file)?);
        }
        Ok(verdicts)
    }

    /// The routed site at one layer over the seeded tokens: per token the
    /// op-path gemvs, the kernel and its rerun, the down on the seeded h,
    /// each against the host transcription of our rule.
    fn routed_set(
        cx: &Cx,
        ly: &Layer,
        input: &Input,
        refs: &[SlotRefs],
        stacks: &Stacks,
        dv: &mut Dev,
    ) -> Result<bool, GateError> {
        let stream = cx.gpu.stream();
        let (k, ff, l) = (cx.meta.n_embd, cx.meta.n_ff, ly.l);
        let limit = cx.meta.clamp_exp[l];
        let mut h_exact = true;
        let (mut g_rel, mut u_rel, mut d_rel) = (0.0f32, 0.0f32, 0.0f32);
        let mut rerun = true;
        let mut hits = [0usize; 3];
        for tt in 0..TOKENS {
            let sel_dev = DeviceBuffer::from_host(stream, &stacks.sel(&input.ids, tt))?;
            dv.x.copy_from_host(stream, &input.x[tt * k..(tt + 1) * k])?;
            cx.gpu.enqueue_quantize_q8_1(&dv.x, &mut dv.act_x)?;
            cx.gpu
                .enqueue_gemv_q3k_sel(&stacks.wg, &dv.act_x, &sel_dev, N_USED, ff, &mut dv.g6)?;
            cx.gpu
                .enqueue_gemv_q3k_sel(&stacks.wu, &dv.act_x, &sel_dev, N_USED, ff, &mut dv.u6)?;
            let args = ExpertGateUp {
                wg: &stacks.wg,
                wu: &stacks.wu,
                act: &dv.act_x,
                sel: &sel_dev,
                n_slots: N_USED,
                rows_per_expert: ff,
                limit,
                fault: cx.gpu.unlabelled_sink(),
            };
            cx.experts
                .enqueue_expert_gate_up(stream, &args, &mut dv.h6)?;
            let (g_op, u_op) = (dv.g6.to_host_vec(stream)?, dv.u6.to_host_vec(stream)?);
            let hk = dv.h6.to_host_vec(stream)?;
            cx.experts
                .enqueue_expert_gate_up(stream, &args, &mut dv.h6)?;
            rerun &= bits_equal(&hk, &dv.h6.to_host_vec(stream)?);

            let h_in = &input.h[tt * N_USED * ff..(tt + 1) * N_USED * ff];
            dv.hin.copy_from_host(stream, h_in)?;
            cx.gpu.enqueue_quantize_q8_1(&dv.hin, &mut dv.act_h)?;
            let q4 = cx.gpu.q4k_sel();
            q4.enqueue_gemv_q4k_sel(
                stream, &stacks.wd, &dv.act_h, &sel_dev, N_USED, k, &mut dv.d6,
            )?;
            let dk = dv.d6.to_host_vec(stream)?;
            q4.enqueue_gemv_q4k_sel(
                stream, &stacks.wd, &dv.act_h, &sel_dev, N_USED, k, &mut dv.d6,
            )?;
            rerun &= bits_equal(&dk, &dv.d6.to_host_vec(stream)?);

            let h_host: Vec<f32> = g_op
                .iter()
                .zip(&u_op)
                .map(|(&g, &u)| swiglu_clamp(g, u, limit))
                .collect();
            h_exact &= bits_equal(&hk, &h_host);
            let (cs, cu, cl) = crossings(&g_op, &u_op, limit);
            hits = [hits[0] + cs, hits[1] + cu, hits[2] + cl];
            for (s, r) in refs[tt * N_USED..(tt + 1) * N_USED].iter().enumerate() {
                let (a, b) = (s * ff, (s + 1) * ff);
                g_rel = g_rel.max(max_rel_err(&g_op[a..b], &r.g)?);
                u_rel = u_rel.max(max_rel_err(&u_op[a..b], &r.u)?);
                let (a, b) = (s * k, (s + 1) * k);
                d_rel = d_rel.max(max_rel_err(&dk[a..b], &r.d)?);
            }
            // PIN(2026-11-18): the real-file ik set retired; ik cannot read bq3km's engram schema; no fixture set carries this clause
        }
        let pass = h_exact
            && g_rel <= KERNEL_BAND
            && u_rel <= KERNEL_BAND
            && d_rel <= KERNEL_BAND
            && rerun;
        println!(
            "routed layer={l} tokens={TOKENS} experts={} limit={limit} clamp_hits={}/{}/{} \
             h_eq_host={h_exact} gate_rel={g_rel:.3e} up_rel={u_rel:.3e} down_rel={d_rel:.3e} \
             rerun={rerun} {}",
            stacks.experts.len(),
            hits[0],
            hits[1],
            hits[2],
            verdict(pass)
        );
        Ok(pass)
    }

    /// `cores::q3_slot` (crates/gpu/src/cores.rs): the q3 plane's u64
    /// pair slot of value-order word `v4`; bit 3 of `v4` picks the half.
    /// Same copy as the hc gate's, one binary to a table.
    fn card_q3_slot(v4: usize) -> usize {
        64 * (v4 >> 7)
            + 32 * ((v4 >> 4) & 1)
            + 16 * ((v4 >> 6) & 1)
            + 8 * ((v4 >> 5) & 1)
            + (v4 & 7)
    }

    /// One column's card q8_1 planes against the host quantizer's bytes:
    /// every block scale's bits and every code word, read back through the
    /// q3 permutation's slots (the q4 plane holds the same words in its own
    /// permutation, and the s8 plane is the exact signed sum of the codes
    /// these words already pin, so both stay covered by this one). The
    /// q3/d8 spans are the column's own.
    fn card_act_bits(host: &qdot::CardQ81, q3: &[u64], d8: &[f32], k: usize) -> (bool, String) {
        // The q3 plane holds 64*ceil(n_sb/2) u64 a column (an odd tail's
        // group is allocated but never written).
        let col_slots = 64 * (k / 256).div_ceil(2);
        if q3.len() < col_slots {
            return (false, format!("short q3 plane: {} < {col_slots}", q3.len()));
        }
        if d8.len() < host.d8().len()
            || d8
                .iter()
                .zip(host.d8())
                .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            let at = d8
                .iter()
                .zip(host.d8())
                .position(|(a, b)| a.to_bits() != b.to_bits())
                .map_or("short plane".to_string(), |i| format!("block {i}"));
            return (false, format!("d8 {at}"));
        }
        for v4 in 0..k / 4 {
            let mut word = 0u32;
            for j in 0..4 {
                word |= ((host.codes()[4 * v4 + j] as u32) & 0xff) << (8 * j);
            }
            let slot = card_q3_slot(v4);
            let pair = q3[slot];
            let got = if v4 & 8 == 0 {
                pair as u32
            } else {
                (pair >> 32) as u32
            };
            if got != word {
                return (
                    false,
                    format!("word {v4}: card {got:#010x} host {word:#010x}"),
                );
            }
        }
        (true, String::new())
    }

    /// The `cardrule` site at one layer over the seeded tokens: the card's own
    /// q8_1 bytes of x and of the seeded h, the fused `ds41_expert_gate_up`'s h
    /// and `q4k_gemv_sel`'s down — against `qdot`'s card-rule kernels on the
    /// same inputs, bit for bit, every token. The first mismatch of each
    /// comparison prints its place and both values. The host dots run on
    /// the pool's width (`par_chunks`), a row a chunk.
    fn cardrule_set(
        cx: &Cx,
        ly: &Layer,
        inp: &Input,
        stacks: &Stacks,
        file: &RoutedFile<'_>,
        dv: &mut Dev,
    ) -> Result<bool, GateError> {
        let stream = cx.gpu.stream();
        let (k, ff, l) = (cx.meta.n_embd, cx.meta.n_ff, ly.l);
        let limit = cx.meta.clamp_exp[l];
        let rb_gu = row_bytes(GgmlType::Q3_K, k)?;
        let rb_d = row_bytes(GgmlType::Q4_K, ff)?;
        let (mut x_ok, mut h_ok, mut hq_ok, mut d_ok) = (true, true, true, true);
        let (mut x_first, mut h_first, mut hq_first, mut d_first) = (
            "-".to_string(),
            "-".to_string(),
            "-".to_string(),
            "-".to_string(),
        );
        for tt in 0..TOKENS {
            let sel = stacks.sel(&inp.ids, tt);
            if sel.iter().any(|&p| p as usize >= stacks.experts.len()) {
                return Err(format!(
                    "cardrule: token {tt} layer {l}: a token's expert is outside the uploaded stacks"
                )
                .into());
            }
            let sel_dev = DeviceBuffer::from_host(stream, &sel)?;
            let ids: Vec<usize> = sel
                .iter()
                .map(|&p| stacks.experts[p as usize] as usize)
                .collect();
            // x: the card quantizer's planes against the host's bytes.
            dv.x.copy_from_host(stream, &inp.x[tt * k..(tt + 1) * k])?;
            cx.gpu.enqueue_quantize_q8_1(&dv.x, &mut dv.act_x)?;
            let xact = qdot::card_q8_1(&inp.x[tt * k..(tt + 1) * k]);
            let (ok, first) = card_act_bits(
                &xact,
                &dv.act_x.q3().to_host_vec(stream)?,
                &dv.act_x.d8().to_host_vec(stream)?,
                k,
            );
            if !ok {
                x_ok = false;
                if x_first == "-" {
                    x_first = format!("token {tt} {first}");
                }
            }
            // The fused gate·up over the card's own q8_1 of x.
            let args = ExpertGateUp {
                wg: &stacks.wg,
                wu: &stacks.wu,
                act: &dv.act_x,
                sel: &sel_dev,
                n_slots: N_USED,
                rows_per_expert: ff,
                limit,
                fault: cx.gpu.unlabelled_sink(),
            };
            cx.experts
                .enqueue_expert_gate_up(stream, &args, &mut dv.h6)?;
            let hk = dv.h6.to_host_vec(stream)?;
            let mut h_host = vec![0.0f32; N_USED * ff];
            {
                let xact = &xact;
                let (gate, up) = (file.gate, file.up);
                par_chunks(&mut h_host, 1, |i, o| {
                    let (s, r) = (i / ff, i % ff);
                    let e = ids[s];
                    let g =
                        qdot::card_q3k_dot_row(&gate[e * ff * rb_gu + r * rb_gu..][..rb_gu], xact)
                            .map_err(|e| e.to_string())?;
                    let u =
                        qdot::card_q3k_dot_row(&up[e * ff * rb_gu + r * rb_gu..][..rb_gu], xact)
                            .map_err(|e| e.to_string())?;
                    o[0] = qdot::card_swiglu_clamp_1(g, u, limit);
                    Ok(())
                })?;
            }
            for s in 0..N_USED {
                for r in 0..ff {
                    let i = s * ff + r;
                    if hk[i].to_bits() != h_host[i].to_bits() {
                        h_ok = false;
                        if h_first == "-" {
                            h_first = format!(
                                "token {tt} slot {s} row {r}: card {:?} ({:#010x}) host {:?} ({:#010x})",
                                hk[i],
                                hk[i].to_bits(),
                                h_host[i],
                                h_host[i].to_bits()
                            );
                        }
                    }
                }
            }
            // The seeded h quantized (the card down's own input), then the
            // sel gemv against the host down.
            let h_d = &inp.h[tt * N_USED * ff..(tt + 1) * N_USED * ff];
            dv.hin.copy_from_host(stream, h_d)?;
            cx.gpu.enqueue_quantize_q8_1(&dv.hin, &mut dv.act_h)?;
            let q4 = cx.gpu.q4k_sel();
            q4.enqueue_gemv_q4k_sel(
                stream, &stacks.wd, &dv.act_h, &sel_dev, N_USED, k, &mut dv.d6,
            )?;
            let dk = dv.d6.to_host_vec(stream)?;
            let (q3h, d8h) = (
                dv.act_h.q3().to_host_vec(stream)?,
                dv.act_h.d8().to_host_vec(stream)?,
            );
            let col_slots = 64 * (ff / 256).div_ceil(2);
            let n_sb = ff / 256;
            let cols: Vec<qdot::CardQ81> = (0..N_USED)
                .map(|s| qdot::card_q8_1(&h_d[s * ff..(s + 1) * ff]))
                .collect();
            for s in 0..N_USED {
                let (ok, first) = card_act_bits(
                    &cols[s],
                    &q3h[s * col_slots..(s + 1) * col_slots],
                    &d8h[s * 2 * n_sb..(s + 1) * 2 * n_sb],
                    ff,
                );
                if !ok {
                    hq_ok = false;
                    if hq_first == "-" {
                        hq_first = format!("token {tt} slot {s} {first}");
                    }
                }
            }
            let mut d_host = vec![0.0f32; N_USED * k];
            {
                let down = file.down;
                let cols = &cols;
                par_chunks(&mut d_host, 1, |i, o| {
                    let (s, r) = (i / k, i % k);
                    let e = ids[s];
                    o[0] =
                        qdot::card_q4k_dot_row(&down[e * k * rb_d + r * rb_d..][..rb_d], &cols[s])
                            .map_err(|e| e.to_string())?;
                    Ok(())
                })?;
            }
            for s in 0..N_USED {
                for r in 0..k {
                    let i = s * k + r;
                    if dk[i].to_bits() != d_host[i].to_bits() {
                        d_ok = false;
                        if d_first == "-" {
                            d_first = format!(
                                "token {tt} slot {s} row {r}: card {:?} ({:#010x}) host {:?} ({:#010x})",
                                dk[i],
                                dk[i].to_bits(),
                                d_host[i],
                                d_host[i].to_bits()
                            );
                        }
                    }
                }
            }
        }
        stream.synchronize()?;
        let fault = cx.gpu.take_fault()?;
        let pass = x_ok && h_ok && hq_ok && d_ok && fault.is_none();
        println!(
            "cardrule layer={l} tokens={TOKENS} experts={} limit={limit} \
             x_bits={x_ok}({x_first}) h_bits={h_ok}({h_first}) hquant_bits={hq_ok}({hq_first}) \
             down_bits={d_ok}({d_first}) fault={fault:?} {}",
            stacks.experts.len(),
            verdict(pass)
        );
        Ok(pass)
    }

    /// The `cardrule` rounding site, once a gate run: a column whose every
    /// 128-value block holds 127 (scale exactly 1) and the values `±(j +
    /// 0.5)`, `±j` and `±(j + 0.25)` — every quotient a tie or an exact
    /// integer, where a half-even rounding and a half-away one part (seeded
    /// columns land on a tie too rarely to pin the mode) — through
    /// the card's quantizer and `qdot::card_q8_1`, bit for bit.
    fn cardrule_ties(cx: &Cx, dv: &mut Dev) -> Result<bool, GateError> {
        let stream = cx.gpu.stream();
        let k = cx.meta.n_embd;
        let col: Vec<f32> = (0..k)
            .map(|i| {
                let j = i % 128;
                let sign = if (i / 128) % 2 == 0 { 1.0 } else { -1.0 };
                match j {
                    0 => 127.0,
                    _ => {
                        let m = ((j - 1) / 3) as f32;
                        sign * match (j - 1) % 3 {
                            0 => m + 0.5,
                            1 => m,
                            _ => m + 0.25,
                        }
                    }
                }
            })
            .collect();
        dv.x.copy_from_host(stream, &col)?;
        stream.synchronize()?;
        cx.gpu.clear_fault()?;
        cx.gpu.enqueue_quantize_q8_1(&dv.x, &mut dv.act_x)?;
        let host = qdot::card_q8_1(&col);
        let (ok, first) = card_act_bits(
            &host,
            &dv.act_x.q3().to_host_vec(stream)?,
            &dv.act_x.d8().to_host_vec(stream)?,
            k,
        );
        stream.synchronize()?;
        let fault = cx.gpu.take_fault()?;
        let ties = col.iter().filter(|v| v.fract().abs() == 0.5).count();
        let pass = ok && fault.is_none();
        println!(
            "cardrule ties k={k} ties={ties} x_bits={ok}({}) fault={fault:?} {}",
            if ok { "-" } else { first.as_str() },
            verdict(pass)
        );
        Ok(pass)
    }

    /// The `cardrule` tile site, once a gate run: the grouped prompt path
    /// (`ds41_expert_gate_up_tiles`, `q4k_gemv_tiles`) over two tokens that
    /// list the same experts — every expert serves two columns, `m = 2` —
    /// against the host card-rule kernels' one-column calls, bit for bit.
    fn cardrule_tiles(
        cx: &Cx,
        ly: &Layer,
        stacks: &Stacks,
        file: &RoutedFile<'_>,
    ) -> Result<bool, GateError> {
        let stream = cx.gpu.stream();
        let (k, ff, l) = (cx.meta.n_embd, cx.meta.n_ff, ly.l);
        let limit = cx.meta.clamp_exp[l];
        let rb_gu = row_bytes(GgmlType::Q3_K, k)?;
        let rb_d = row_bytes(GgmlType::Q4_K, ff)?;
        if stacks.experts.len() < N_USED {
            return Err("cardrule tiles: fewer uploaded experts than slots".into());
        }
        // Two tokens over the first N_USED places: token t's slot e + t*N_USED
        // reads place e, so entry order groups each expert's two columns.
        // The table covers every place the stacks hold (the launcher counts
        // the stack's experts); the places past the first N_USED get empty
        // runs.
        let n_tbl = stacks.experts.len();
        let slots = 2 * N_USED;
        let order: Vec<u32> = (0..N_USED)
            .flat_map(|e| [e as u32, (e + N_USED) as u32])
            .collect();
        let start: Vec<u32> = (0..=n_tbl).map(|e| (2 * e.min(N_USED)) as u32).collect();
        let x = activations(k, 2, 0xE0A1_1757);
        let x_ord: Vec<f32> = (0..slots)
            .flat_map(|j| {
                let t = order[j] as usize / N_USED;
                x[t * k..(t + 1) * k].to_vec()
            })
            .collect();
        let x_ord = DeviceBuffer::from_host(stream, &x_ord)?;
        let mut act_ord = Q8Act::with_slots(stream, slots, k)?;
        cx.gpu.enqueue_quantize_q8_1(&x_ord, &mut act_ord)?;
        let start_dev = DeviceBuffer::from_host(stream, &start)?;
        let order_dev = DeviceBuffer::from_host(stream, &order)?;
        let mut tiles = DeviceBuffer::from_host(stream, &vec![0u32; tile_cap(slots, n_tbl) + 1])?;
        let mut h = DeviceBuffer::from_host(stream, &vec![f32::NAN; slots * ff])?;
        let kernels = FfnBatchKernels::load(cx.gpu.context())?;
        stream.synchronize()?;
        cx.gpu.clear_fault()?;
        cx.gpu.q4k_sel().enqueue_grouped_tiles(
            stream,
            &start_dev,
            n_tbl,
            slots,
            cx.gpu.unlabelled_sink(),
            &mut tiles,
        )?;
        let g = TiledGateUp {
            wg: stacks.wg.buf(),
            wu: stacks.wu.buf(),
            q3: act_ord.q3(),
            d8: act_ord.d8(),
            start: &start_dev,
            tiles: &tiles,
            n_experts: n_tbl,
            rows_per_expert: ff,
            n_slots: slots,
            n_sb: k / 256,
            limit,
        };
        kernels.enqueue_gate_up_tiles(stream, &g, cx.gpu.unlabelled_sink(), &mut h)?;
        let h = h; // entry-major: entry j is slot order[j]'s
        let mut act_h = Q8Act::with_slots(stream, slots, ff)?;
        cx.gpu.enqueue_quantize_q8_1(&h, &mut act_h)?;
        let table = TileTable {
            order: &order_dev,
            start: &start_dev,
            tiles: &tiles,
            n_experts: n_tbl,
            n_slots: slots,
        };
        let down = TiledDown {
            w: &stacks.wd,
            act: &act_h,
            table,
            rows_per_expert: k,
        };
        let mut y = DeviceBuffer::from_host(stream, &vec![f32::NAN; slots * k])?;
        cx.gpu
            .q4k_sel()
            .enqueue_gemv_q4k_tiles(stream, &down, cx.gpu.unlabelled_sink(), &mut y)?;
        let (hv, yv) = (h.to_host_vec(stream)?, y.to_host_vec(stream)?);
        stream.synchronize()?;
        let fault = cx.gpu.take_fault()?;
        // The host side, one column a time: each token's x, each expert's
        // gate·up·SwiGLU rows, its h quantized, its down rows.
        let mut want_h = vec![0.0f32; slots * ff];
        let mut want_y = vec![0.0f32; slots * k];
        for t in 0..2 {
            let xcol = qdot::card_q8_1(&x[t * k..(t + 1) * k]);
            for e in 0..N_USED {
                let e_id = stacks.experts[e] as usize;
                let base = e_id * ff * rb_gu;
                let mut hcol = vec![0.0f32; ff];
                for (r, o) in hcol.iter_mut().enumerate() {
                    let g = qdot::card_q3k_dot_row(&file.gate[base + r * rb_gu..][..rb_gu], &xcol)?;
                    let u = qdot::card_q3k_dot_row(&file.up[base + r * rb_gu..][..rb_gu], &xcol)?;
                    *o = qdot::card_swiglu_clamp_1(g, u, limit);
                }
                // The tile gate·up's entry for (expert e, token t) is entry
                // 2e + t of the grouped order; the down's slot is
                // e + t*N_USED.
                want_h[(2 * e + t) * ff..(2 * e + t + 1) * ff].copy_from_slice(&hcol);
                let q = qdot::card_q8_1(&hcol);
                let base_d = e_id * k * rb_d;
                for (r, o) in want_y[(e + t * N_USED) * k..][..k].iter_mut().enumerate() {
                    *o = qdot::card_q4k_dot_row(&file.down[base_d + r * rb_d..][..rb_d], &q)?;
                }
            }
        }
        let (h_ok, h_first) = bits_first(&hv, &want_h, "entry");
        let (y_ok, y_first) = bits_first(&yv, &want_y, "slot");
        let pass = h_ok && y_ok && fault.is_none();
        println!(
            "cardrule tiles layer={l} tokens=2 experts={N_USED} columns_per_expert=2 \
             h_bits={h_ok}({h_first}) down_bits={y_ok}({y_first}) fault={fault:?} {}",
            verdict(pass)
        );
        Ok(pass)
    }

    /// The first bit mismatch of two equal-length f32 slices, as (ok, place).
    fn bits_first(got: &[f32], want: &[f32], what: &str) -> (bool, String) {
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            if a.to_bits() != b.to_bits() {
                return (
                    false,
                    format!(
                        "{what} {i}: card {a:?} ({:#010x}) host {b:?} ({:#010x})",
                        a.to_bits(),
                        b.to_bits()
                    ),
                );
            }
        }
        (true, String::new())
    }

    /// The smallest power of two `c` with `c·v >= target` (at least 1): a
    /// power-of-two scale moves every q8_1 code by none and every dot by
    /// exactly `c`.
    fn pow2_reaching(v: f32, target: f32) -> f32 {
        if v.is_nan() || v <= 0.0 {
            return 1.0;
        }
        let mut c = 1.0f32;
        while c * v < target && c < 1e30 {
            c *= 2.0;
        }
        c
    }

    /// Counts of the clamp's three crossings over rows of `g` and `u`:
    /// `silu(g) > L`, `u > L`, `u < -L`.
    fn crossings(g: &[f32], u: &[f32], limit: f32) -> (usize, usize, usize) {
        let s = g.iter().filter(|&&v| silu_ik(v) > limit).count();
        (
            s,
            u.iter().filter(|&&v| v > limit).count(),
            u.iter().filter(|&&v| v < -limit).count(),
        )
    }

    /// The largest ratio of a kernel's `h` to the clamp's statement, the f64
    /// `swiglu64` at the kernel's own f32 dots: within `SILU_ROUNDING` of the
    /// value (a clamp is exact and moves no value further than its input
    /// moved), and `HALF_SUBNORMAL` for a product below the normal range.
    /// Where `e^-g`, within its 3.92u (rounded up to 4u), can pass
    /// `f32::MAX`, ik's silu may be -0, so there the value itself is the band.
    fn stated_ratio(h: &[f32], g: &[f32], u: &[f32], limit: f32) -> f64 {
        h.iter().zip(g).zip(u).fold(0.0, |r, ((&k, &gv), &uv)| {
            let g64 = f64::from(gv);
            let e = swiglu64(g64, f64::from(uv), limit);
            let flush = (-g64).exp() * (1.0 + 4.0 * U) >= f64::from(f32::MAX);
            let rel = if flush {
                1.0 + SILU_ROUNDING
            } else {
                SILU_ROUNDING
            };
            worst(r, (f64::from(k) - e).abs(), rel * e.abs() + HALF_SUBNORMAL)
        })
    }

    /// The routed kernel with its input scaled until the clamp bites on all
    /// three sides, at the layer's limit and at 0 (no clamp), against the
    /// host on the op path's dots and against the clamp's statement.
    fn clamp_routed(
        cx: &Cx,
        dv: &mut Dev,
        s: &Stacks,
        sel: &[u32],
        x: &[f32],
        l: usize,
    ) -> Result<bool, GateError> {
        let stream = cx.gpu.stream();
        let ff = cx.meta.n_ff;
        let limit = cx.meta.clamp_exp[l];
        let sel_dev = DeviceBuffer::from_host(stream, sel)?;
        let run = |dv: &mut Dev, x: &[f32]| -> Result<(Vec<f32>, Vec<f32>), GateError> {
            dv.x.copy_from_host(stream, x)?;
            cx.gpu.enqueue_quantize_q8_1(&dv.x, &mut dv.act_x)?;
            cx.gpu
                .enqueue_gemv_q3k_sel(&s.wg, &dv.act_x, &sel_dev, N_USED, ff, &mut dv.g6)?;
            cx.gpu
                .enqueue_gemv_q3k_sel(&s.wu, &dv.act_x, &sel_dev, N_USED, ff, &mut dv.u6)?;
            Ok((dv.g6.to_host_vec(stream)?, dv.u6.to_host_vec(stream)?))
        };
        let (g1, u1) = run(dv, x)?;
        let top = |v: &[f32], sign: f32| v.iter().fold(0.0f32, |m, &a| m.max(sign * a));
        let c = [top(&g1, 1.0), top(&u1, 1.0), top(&u1, -1.0)]
            .iter()
            .map(|&v| pow2_reaching(v, 2.0 * limit))
            .fold(1.0f32, f32::max);
        let xs: Vec<f32> = x.iter().map(|&v| v * c).collect();
        let (g, u) = run(dv, &xs)?;
        let (mut ok, mut stated) = (true, 0.0f64);
        for lim in [limit, 0.0] {
            let args = ExpertGateUp {
                wg: &s.wg,
                wu: &s.wu,
                act: &dv.act_x,
                sel: &sel_dev,
                n_slots: N_USED,
                rows_per_expert: ff,
                limit: lim,
                fault: cx.gpu.unlabelled_sink(),
            };
            cx.experts
                .enqueue_expert_gate_up(stream, &args, &mut dv.h6)?;
            let hk = dv.h6.to_host_vec(stream)?;
            let host: Vec<f32> = g
                .iter()
                .zip(&u)
                .map(|(&a, &b)| swiglu_clamp(a, b, lim))
                .collect();
            ok &= bits_equal(&hk, &host);
            stated = stated.max(stated_ratio(&hk, &g, &u, lim));
        }
        let (cs, cu, cl) = crossings(&g, &u, limit);
        let pass = ok && stated <= 1.0 && cs > 0 && cu > 0 && cl > 0;
        println!(
            "clamp_routed layer={l} x_scale={c} limit={limit} silu_g_above={cs} u_above={cu} u_below={cl} \
             kernel_eq_host_at_limit_and_0={ok} stated_ratio={stated:.3} {}",
            verdict(pass)
        );
        Ok(pass)
    }

    /// The shared kernel with its input scaled until the clamp bites on all
    /// three sides, at the layer's limit and at 0, against the host on the
    /// op path's dots and against the clamp's statement.
    fn clamp_shared(cx: &Cx, dv: &mut Dev, ly: &Layer, x: &[f32]) -> Result<bool, GateError> {
        let stream = cx.gpu.stream();
        let limit = cx.meta.clamp_shexp[ly.l];
        let run = |dv: &mut Dev, x: &[f32]| -> Result<(Vec<f32>, Vec<f32>), GateError> {
            dv.x.copy_from_host(stream, x)?;
            if ly.sh_gate.ty == GgmlType::Q3_K {
                cx.gpu.enqueue_quantize_q8_1(&dv.x, &mut dv.act_x)?;
            }
            enqueue_shared_dots(cx, ly, &dv.x, &dv.act_x, &mut dv.sg, &mut dv.su)?;
            Ok((dv.sg.to_host_vec(stream)?, dv.su.to_host_vec(stream)?))
        };
        let (g1, u1) = run(dv, x)?;
        let top = |v: &[f32], sign: f32| v.iter().fold(0.0f32, |m, &a| m.max(sign * a));
        let c = [top(&g1, 1.0), top(&u1, 1.0), top(&u1, -1.0)]
            .iter()
            .map(|&v| pow2_reaching(v, 2.0 * limit))
            .fold(1.0f32, f32::max);
        let xs: Vec<f32> = x.iter().map(|&v| v * c).collect();
        let (g, u) = run(dv, &xs)?;
        let (mut ok, mut stated) = (true, 0.0f64);
        for lim in [limit, 0.0] {
            enqueue_shared_gate_up(cx, ly, &dv.x, &dv.act_x, lim, &mut dv.sh)?;
            let hk = dv.sh.to_host_vec(stream)?;
            let host: Vec<f32> = g
                .iter()
                .zip(&u)
                .map(|(&a, &b)| swiglu_clamp(a, b, lim))
                .collect();
            ok &= bits_equal(&hk, &host);
            stated = stated.max(stated_ratio(&hk, &g, &u, lim));
        }
        let (cs, cu, cl) = crossings(&g, &u, limit);
        let pass = ok && stated <= 1.0 && cs > 0 && cu > 0 && cl > 0;
        println!(
            "clamp_shared layer={} x_scale={c} limit={limit} silu_g_above={cs} u_above={cu} u_below={cl} \
             kernel_eq_host_at_limit_and_0={ok} stated_ratio={stated:.3} {}",
            ly.l,
            verdict(pass)
        );
        Ok(pass)
    }

    /// An id past the resident stack leaves its slot of `h` as it was and
    /// every other slot as the in-range launch writes it.
    fn out_of_range(
        cx: &Cx,
        dv: &mut Dev,
        s: &Stacks,
        sel: &[u32],
        x: &[f32],
        limit: f32,
    ) -> Result<bool, GateError> {
        const BAD_SLOT: usize = 3;
        let stream = cx.gpu.stream();
        let ff = cx.meta.n_ff;
        dv.x.copy_from_host(stream, x)?;
        cx.gpu.enqueue_quantize_q8_1(&dv.x, &mut dv.act_x)?;
        let good = DeviceBuffer::from_host(stream, sel)?;
        let mut bad_sel = sel.to_vec();
        bad_sel[BAD_SLOT] = u32::try_from(s.experts.len())?;
        let bad = DeviceBuffer::from_host(stream, &bad_sel)?;
        let launch = |dv: &mut Dev, sel: &DeviceBuffer<u32>| -> Result<Vec<f32>, GateError> {
            let args = ExpertGateUp {
                wg: &s.wg,
                wu: &s.wu,
                act: &dv.act_x,
                sel,
                n_slots: N_USED,
                rows_per_expert: ff,
                limit,
                fault: cx.gpu.unlabelled_sink(),
            };
            cx.experts
                .enqueue_expert_gate_up(stream, &args, &mut dv.h6)?;
            Ok(dv.h6.to_host_vec(stream)?)
        };
        let h_good = launch(dv, &good)?;
        let sentinel = f32::from_bits(0x7fc0_0bad);
        dv.h6.copy_from_host(stream, &vec![sentinel; N_USED * ff])?;
        let h_bad = launch(dv, &bad)?;
        let untouched = h_bad[BAD_SLOT * ff..(BAD_SLOT + 1) * ff]
            .iter()
            .all(|v| v.to_bits() == sentinel.to_bits());
        let others = (0..N_USED)
            .filter(|&j| j != BAD_SLOT)
            .all(|j| bits_equal(&h_bad[j * ff..(j + 1) * ff], &h_good[j * ff..(j + 1) * ff]));
        let pass = untouched && others;
        println!(
            "oor sel[{BAD_SLOT}]={} (resident {}) bad_slot_untouched={untouched} good_slots_bit_identical={others} {}",
            bad_sel[BAD_SLOT],
            s.experts.len(),
            verdict(pass)
        );
        Ok(pass)
    }

    /// Enqueue one layer's seven launches for the token in `dv.x`: the router,
    /// the q8_1 of `x`, the routed gate·up·SwiGLU, the q8_1 of `h`, the
    /// routed down, the shared gate·up·SwiGLU (a Q3_K one on the q8_1 of
    /// `x`), the shared down — after the q8_1 of its input where the weight
    /// reads one, the eighth. The routed
    /// slots read `sel` (the seeded tokens' ids in the compact stack), not the
    /// router's ids: the gate's stacks hold only the experts those tokens use.
    fn layer_chain(
        cx: &Cx,
        dv: &mut Dev,
        ly: &Layer,
        s: &Stacks,
        sel: &DeviceBuffer<u32>,
    ) -> Result<(), bloomery_gpu::GpuError> {
        let stream = cx.gpu.stream();
        let (k, ff) = (cx.meta.n_embd, cx.meta.n_ff);
        cx.router.enqueue_router(
            stream,
            &ly.router_dev,
            &dv.x,
            &ly.bias_dev,
            cx.meta.scale,
            &mut dv.rout,
            cx.gpu.unlabelled_sink(),
        )?;
        cx.gpu.enqueue_quantize_q8_1(&dv.x, &mut dv.act_x)?;
        let args = ExpertGateUp {
            wg: &s.wg,
            wu: &s.wu,
            act: &dv.act_x,
            sel,
            n_slots: N_USED,
            rows_per_expert: ff,
            limit: cx.meta.clamp_exp[ly.l],
            fault: cx.gpu.unlabelled_sink(),
        };
        cx.experts
            .enqueue_expert_gate_up(stream, &args, &mut dv.h6)?;
        cx.gpu.enqueue_quantize_q8_1(&dv.h6, &mut dv.act_h)?;
        cx.gpu
            .q4k_sel()
            .enqueue_gemv_q4k_sel(stream, &s.wd, &dv.act_h, sel, N_USED, k, &mut dv.d6)?;
        let limit = cx.meta.clamp_shexp[ly.l];
        enqueue_shared_gate_up(cx, ly, &dv.x, &dv.act_x, limit, &mut dv.sh)?;
        enqueue_shared_down(cx, ly, &dv.sh, &mut dv.act_sh, &mut dv.sy)
    }

    /// Every buffer the layer chain writes, read back.
    fn chain_outputs(dv: &Dev, stream: &CudaStream) -> Result<Vec<Vec<f32>>, GateError> {
        let ids: Vec<f32> = dv
            .rout
            .ids
            .to_host_vec(stream)?
            .iter()
            .map(|&i| f32::from_bits(i))
            .collect();
        Ok(vec![
            dv.rout.logits.to_host_vec(stream)?,
            dv.rout.probs.to_host_vec(stream)?,
            ids,
            dv.rout.weights.to_host_vec(stream)?,
            dv.h6.to_host_vec(stream)?,
            dv.d6.to_host_vec(stream)?,
            dv.sh.to_host_vec(stream)?,
            dv.sy.to_host_vec(stream)?,
        ])
    }

    /// One layer's launches captured and replayed twice: seven graph nodes
    /// (eight where the shared down reads q8_1), each replay's outputs
    /// bit-identical to the eager run's, and the router's ticket count back
    /// at zero after each.
    fn graph(
        cx: &Cx,
        dv: &mut Dev,
        ly: &Layer,
        s: &Stacks,
        sel: &[u32],
        x: &[f32],
    ) -> Result<bool, GateError> {
        // PIN(2026-09-24): 8 → 7 — the layer chain's combine launch left with `ds41_moe_combine`.
        const NODES: usize = 7;
        let want = NODES + usize::from(ly.sh_down.dense()?.reads_q8_1());
        let stream = cx.gpu.stream();
        let sel = DeviceBuffer::from_host(stream, sel)?;
        dv.x.copy_from_host(stream, x)?;
        layer_chain(cx, dv, ly, s, &sel)?;
        let eager = chain_outputs(dv, stream)?;
        let mut tickets_zero = dv.rout.tickets(stream)? == 0;
        let g = cx.gpu.capture(|_| layer_chain(cx, dv, ly, s, &sel))?;
        let nodes = g.node_count();
        let mut same = true;
        for _ in 0..2 {
            g.launch(stream)?;
            stream.synchronize()?;
            let out = chain_outputs(dv, stream)?;
            same &= out.iter().zip(&eager).all(|(a, b)| bits_equal(a, b));
            tickets_zero &= dv.rout.tickets(stream)? == 0;
        }
        let pass = nodes == want && same && tickets_zero;
        println!(
            "graph layer={} nodes={nodes} (want {want}) replays_eq_eager={same} tickets_zero={tickets_zero} {}",
            ly.l,
            verdict(pass)
        );
        Ok(pass)
    }

    /// Fewer finite candidates than slots: a K = 32 router whose weight rows
    /// are NaN except those of four experts (`x` finite and one-hot), so
    /// every other expert's logit, score and selection value is NaN and
    /// `take` accepts none of them. Rounds five and six have no winner. The
    /// kernel must raise [`FaultSite::Router`]; before the fault word the
    /// rounds left `(−inf, 0)` and wrote expert 0 again with its finite
    /// weight — duplicate ids, every weight finite, nothing any probe sees.
    /// The line prints what the kernel wrote either way.
    fn router_nan(cx: &Cx, dv: &mut Dev) -> Result<bool, GateError> {
        const K: usize = 32;
        // Expert 0 among the four: rounds five and six fall back to id 0,
        // whose score is finite, so the old kernel's weights came out finite.
        const LIVE: [usize; 4] = [0, 42, 100, 300];
        let stream = cx.gpu.stream();
        let mut w = vec![f32::NAN; N_EXPERT * K];
        for (j, &r) in LIVE.iter().enumerate() {
            w[r * K..(r + 1) * K].fill(0.0);
            w[r * K] = 5.0 + j as f32;
        }
        let mut x = vec![0.0f32; K];
        x[0] = 1.0;
        let w_dev = DeviceTensor::upload(stream, &w, N_EXPERT, K)?;
        let x_dev = DeviceBuffer::from_host(stream, &x)?;
        let b_dev = DeviceBuffer::from_host(stream, &vec![0.0f32; N_EXPERT])?;
        cx.router.enqueue_router(
            stream,
            &w_dev,
            &x_dev,
            &b_dev,
            cx.meta.scale,
            &mut dv.rout,
            cx.gpu.unlabelled_sink(),
        )?;
        let ids = dv.rout.ids.to_host_vec(stream)?;
        let wk = dv.rout.weights.to_host_vec(stream)?;
        let tickets = dv.rout.tickets(stream)?;
        let got = cx.gpu.take_fault()?;
        let want = Fault::at(LAYER_NONE, FaultSite::Router);
        let mut distinct = ids.clone();
        distinct.sort_unstable();
        distinct.dedup();
        let pass = got == Some(want) && tickets == 0;
        println!(
            "nan_rows case=four_finite_experts ids={ids:?} distinct={} weights_finite={} tickets={tickets} \
             want=router got={} {}",
            distinct.len(),
            wk.iter().all(|v| v.is_finite()),
            got.map_or("none".to_string(), |f| f.to_string()),
            verdict(pass)
        );
        Ok(pass)
    }

    /// The router's selection on planted ties (a K = 32 router whose logits
    /// are its first weight column against a one-hot input), against the
    /// rule's statement and the host's selection from the kernel's scores;
    /// the weights finite in every case, `all_underflow` too, whose every
    /// score is 0 (a logit of −30 is below the −16.6 where `1 + e^x` rounds
    /// to 1).
    fn router_ties(cx: &Cx, dv: &mut Dev) -> Result<bool, GateError> {
        const K: usize = 32;
        let stream = cx.gpu.stream();
        let scale = cx.meta.scale;
        let base: Vec<f32> = (0..N_EXPERT).map(|r| -3.0 + 0.01 * r as f32).collect();
        let planted = |pairs: &[(usize, f32)]| -> Vec<f32> {
            let mut v = base.clone();
            for &(i, val) in pairs {
                v[i] = val;
            }
            v
        };
        let zero = vec![0.0f32; N_EXPERT];
        let mut bias_decides = zero.clone();
        bias_decides[7] = 0.5;
        bias_decides[77] = 0.5;
        let cases: Vec<(&str, Vec<f32>, Vec<f32>)> = vec![
            (
                "one_lane_in_top6",
                planted(&[(10, 10.0), (42, 10.0)]),
                zero.clone(),
            ),
            (
                "boundary_6_7",
                planted(&[
                    (300, 6.0),
                    (301, 6.1),
                    (302, 6.2),
                    (303, 6.3),
                    (304, 6.4),
                    (100, 5.0),
                    (200, 5.0),
                ]),
                zero.clone(),
            ),
            (
                "adjacent_lanes",
                planted(&[(64, 7.0), (65, 7.0)]),
                zero.clone(),
            ),
            ("ends", planted(&[(0, 7.0), (383, 7.0)]), zero.clone()),
            (
                "three_way",
                planted(&[(5, 7.0), (37, 7.0), (200, 7.0)]),
                zero.clone(),
            ),
            ("all_equal", vec![1.0; N_EXPERT], zero.clone()),
            ("bias_decides", vec![1.0; N_EXPERT], bias_decides),
            ("all_underflow", vec![-30.0; N_EXPERT], zero),
        ];
        let mut x = vec![0.0f32; K];
        x[0] = 1.0;
        let x_dev = DeviceBuffer::from_host(stream, &x)?;
        let mut all = true;
        for (name, logits, bias) in cases {
            let mut w = vec![0.0f32; N_EXPERT * K];
            for (r, &v) in logits.iter().enumerate() {
                w[r * K] = v;
            }
            let w_dev = DeviceTensor::upload(stream, &w, N_EXPERT, K)?;
            let b_dev = DeviceBuffer::from_host(stream, &bias)?;
            cx.router.enqueue_router(
                stream,
                &w_dev,
                &x_dev,
                &b_dev,
                scale,
                &mut dv.rout,
                cx.gpu.unlabelled_sink(),
            )?;
            let pk = dv.rout.probs.to_host_vec(stream)?;
            let ids_k = dv.rout.ids.to_host_vec(stream)?;
            let wk = dv.rout.weights.to_host_vec(stream)?;
            let lk = dv.rout.logits.to_host_vec(stream)?;
            // The statement: order by the exact score plus bias, descending,
            // a tie to the larger index. Planted ties are identical inputs,
            // hence identical exact keys; the rest are 0.01 apart in x.
            let key: Vec<f64> = logits
                .iter()
                .zip(&bias)
                .map(|(&l, &b)| softplus64(l) + f64::from(b))
                .collect();
            let stated = top_by(&key, f64::total_cmp);
            let biased: Vec<f32> = pk.iter().zip(&bias).map(|(&p, &b)| p + b).collect();
            let host = select(&biased)?;
            let (w_host, _, _) = weights_rule(&pk, &ids_k, scale);
            let logits_exact = bits_equal(&lk, &logits);
            let finite = wk.iter().all(|w| w.is_finite());
            let tickets = dv.rout.tickets(stream)?;
            let pass = ids_k.as_slice() == stated.as_slice()
                && ids_k.as_slice() == host.as_slice()
                && bits_equal(&wk, &w_host)
                && finite
                && logits_exact
                && tickets == 0;
            println!(
                "ties case={name} ids={ids_k:?} stated={stated:?} host={host:?} weights_exact={} weights_finite={finite} \
                 logits_exact={logits_exact} tickets={tickets} {}",
                bits_equal(&wk, &w_host),
                verdict(pass)
            );
            all &= pass;
        }
        Ok(all)
    }
}
