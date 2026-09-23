//! GPU gate for DeepSeek-V4.1's router and experts — B4 op blocks J and K
//! (`docs/research/v41-b4-plan-report.md` §1-J, §1-K) — against ik's CPU
//! dumps: the 5-token prefill set and the three decode-step sets, every
//! layer, each token launched alone (the decode shape of the kernels).
//!
//! Sites, per layer and set:
//! - `router`: `ds41_router` — `ffn_moe_logits`, `_probs`, `_topk` and
//!   `_weights_scaled` (with ik's `_probs_biased`, `_weights_sum` and
//!   `_weights_norm` in the simulation), every layer;
//! - `routed`: `ds41_expert_gate_up` for `ffn_moe_gate_par` and `q4k_sel`'s
//!   gemv for `ffn_moe_down`, at the layers whose down stack is q4_K — a
//!   q5_K down (layers 0 and 1) is the host tier's in every plan;
//! - `shared`: `ds41_shexp_gate_up` for `ffn_up_gate` and `q8_0_gemv` for
//!   `ffn_shexp`, every layer;
//! - `combine`: `ds41_moe_combine` for `ffn_moe_out` and `ffn_out`, every
//!   layer.
//!
//! Every kernel reads its node's own input from the dump: `ffn_norm`, the
//! dump's expert ids, `ffn_moe_gate_par` for the routed down, `ffn_up_gate`
//! for the shared down, `ffn_moe_down` and the dump's weights for the
//! combine. The routed and shared lines count where the clamp bites on the
//! op path's dots (`clamp_hits`: `silu(g) > L` / `u > L` / `u < -L`); where
//! it does, the dump checks the clamp as well.
//!
//! Three comparisons per site (the B4 gate form):
//! 1. **kernel vs the host transcription of our rule** — bit-identical where
//!    the host runs the rule itself: the SwiGLU on the op-path kernels' dots
//!    (`q3k_gemv_sel`, `q8_0_gemv`: the same bodies and warp tree), the
//!    selection and the weights from the kernel's own scores, the combine;
//!    each dot within `KERNEL_BAND` of its f64 value; the router's logits
//!    bit-identical to `f32_gemv`'s; the scores within the distance of the
//!    device's `expf`/`logf` from the host's (`softplus_err`); every kernel
//!    rerun bit-identical.
//! 2. **ik's rule on the host vs the dump** — the semantics: the scores from
//!    the dump's logits with the host's libm (ik's own), the bias, the
//!    selection, the f64 sum, the division and the scale, the combine and
//!    the final add, all bit for bit; each dot within ik's own accumulation
//!    bound around its f64 value.
//! 3. **kernel vs the dump**, per value, in the band the two rules' difference
//!    gives: the ids exact, the combine bit-identical.
//!
//! The bands. Each output row of a dot is computed here in f64 twice, from
//! the dump's input: with our rule's activations (q8_1 per 128 values for
//! the K-quant stacks, f32 for q8_0 and the router) and with ik's (q8_K per
//! 256 for q3_K, q8_2 per 32 for q4_K and q8_0, f32 for the router), each
//! with its sum of absolute terms `A`. A computed dot lies within `n·u·A` of
//! its f64 value — the standard summation bound, `u` = 2^-24 and `n` the
//! roundings on one term's path (`n_ours`, `n_ik_quant`, `n_ik_f32`) —
//! so the two engines' dots differ by at most `|ours - ik| + (n_ours·A_ours +
//! n_ik·A_ik)·u`. A SwiGLU carries that through the function: the f64
//! SwiGLU at both centres, the slopes (`SILU_SLOPE`, `SILU_FLOOR`, the
//! clamp) times the accumulation radii, and each side's own f32 evaluation
//! (`SILU_ROUNDING`). The scores' band is the two libms' ulp bounds carried
//! through `sqrt(ln(1 + e^x))` (`softplus_err`); the weights', the scores'
//! f64 difference carried through the normalization plus three roundings a
//! side (`WEIGHT_ROUNDING`). A ratio `|difference| / band` above 1 fails.
//!
//! Synthetic checks: router ties planted where the selection treats them
//! differently (one lane, adjacent lanes, the two ends, three ways, the
//! 6th/7th boundary, all equal, the bias deciding), against the rule's
//! statement; the clamp crossed on `silu(g)`, on `u > L` and on `u < -L`,
//! and `L = 0` (no clamp), routed and shared, against the host and against
//! the clamp's statement in f64 (`stated_ratio`); an out-of-range expert id;
//! one layer's eight launches captured as a graph and replayed.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_deepseek41_moe: built without the `deepseek41` feature; see `just gate-gpu-ds41-moe`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_deepseek41_moe", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use bloomery_gpu::weights::{DevWeight, Q8Block, q8_0_planes};
    use bloomery_gpu::{DeviceTensor, Gpu, Q8Act};
    use bloomery_gpu_deepseek41::experts::{ExpertGateUp, ExpertKernels, silu_ik, swiglu_clamp};
    use bloomery_gpu_deepseek41::router::{
        N_EXPERT, N_USED, RouterKernels, RouterOut, sqrt_softplus,
    };
    use bloomery_gpu_gates::oracle::deepseek41::{D1, D2, STEP4};
    use bloomery_gpu_gates::oracle::{self, Set};
    use bloomery_gpu_gates::{
        GateError, KERNEL_BAND, RefManifest, RefRow, bits_equal, bytes_to_words, checks_failed,
        max_rel_err, q8_1_dequant, ref_model_path, ref_tensor_of_in, row_bytes,
        topk_ids_logical_within, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::quant::{
        GgmlType, dequant_row, quantize_activations, quantize_row_q8_2_x4_roundtrip,
    };
    use gguf::{Split, Value};
    use model::arch::Arch;

    /// The decode-step sets this gate reads (T = 1), after the 5-token
    /// prefill set: of the table's (`step_sets`), the fused ones at 4, 301
    /// and 1,025.
    const STEP_SETS: [&str; 3] = [STEP4, D1, D2];

    /// f32's unit roundoff, 2^-24.
    const U: f64 = f32::EPSILON as f64 / 2.0;

    /// Roundings on one term's path through one of our dots over `k` values:
    /// at most one per value a lane reads (the f32 and q8_0 bodies round once
    /// per value, a fused multiply-add each; the K-quant bodies round less,
    /// their integer sums being exact), `k/32` values per lane, then the five
    /// butterfly levels.
    fn n_ours(k: usize) -> f64 {
        (k / 32 + 5) as f64
    }

    /// The same count for ik's quantized dots, whose order this gate does not
    /// transcribe: at most four roundings per 32-value block — the scale
    /// product, the block's multiply-add, and the min term's product and add
    /// (q4_K) — then a horizontal sum of at most five levels.
    fn n_ik_quant(k: usize) -> f64 {
        (4 * (k / 32) + 5) as f64
    }

    /// ik's f32 router dot, its order unread: a sequential sum's bound, one
    /// rounding per value.
    fn n_ik_f32(k: usize) -> f64 {
        k as f64
    }

    /// The f64 references decode q4_K weights as `dequant_row` does, rounding
    /// `d·sc·q - dmin·m` to f32 once; both engines use the exact weight, so
    /// each side's count gains one rounding against this reference. The
    /// other types decode exactly: bf16, q8_0, and q3_K, whose `d·(s - 32)·q`
    /// has at most 20 significant bits.
    const Q4K_DECODE: f64 = 1.0;

    /// silu's steepest slope (1.0998… at x ≈ 2.40), rounded up.
    const SILU_SLOPE: f64 = 1.1;

    /// silu's lowest value in magnitude (-0.27846… at x ≈ -1.2785), rounded
    /// up: `|silu(g)| <= max(|g|, SILU_FLOOR)`.
    const SILU_FLOOR: f64 = 0.2785;

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

    /// One side's weight against the exact `p_s / Σp · scale` of its own
    /// scores, relative: the f64 sum of six (six roundings of 2^-53), its
    /// narrowing to f32, the division and the scale (three roundings of u),
    /// and the second-order terms of those three (below 4u²).
    const WEIGHT_ROUNDING: f64 = 3.0 * U + 6.0 * (f64::EPSILON / 2.0) + 4.0 * U * U;

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
            let want = Arch::Deepseek41.name();
            if split.architecture() != Some(want) {
                return Err(format!(
                    "the model file is {:?}, want {want} — run through `just \
                     gate-gpu-ds41-moe`, which picks the deepseek41 profile",
                    split.architecture()
                )
                .into());
            }
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

    /// A dump set, its manifest read once, and its token count.
    struct SetData {
        name: &'static str,
        man: RefManifest,
        t: usize,
    }

    /// Row `name`/0 of `set`: its type, dims and op proven, and (when given)
    /// its two sources.
    fn row<'a>(
        set: &'a SetData,
        name: &str,
        ne: [u64; 4],
        op: &str,
        srcs: Option<(&str, &str)>,
    ) -> Result<&'a RefRow, GateError> {
        let r = set.man.tensor(name, 0)?;
        r.expect(name, "f32", ne, op)?;
        if let Some((a, b)) = srcs
            && (r.src0.as_deref() != Some(a) || r.src1.as_deref() != Some(b))
        {
            return Err(format!(
                "{}: {name} reads {:?} and {:?}, want {a} and {b}",
                set.name, r.src0, r.src1
            )
            .into());
        }
        Ok(r)
    }

    /// The values of row `name`/0 of `set`, proven as [`row`] proves it.
    fn dump(
        set: &SetData,
        name: &str,
        ne: [u64; 4],
        op: &str,
        srcs: Option<(&str, &str)>,
    ) -> Result<Vec<f32>, GateError> {
        ref_tensor_of_in(&set.man.dir, row(set, name, ne, op, srcs)?)
    }

    /// The expert ids of `set` at layer `l`: the logical twin of
    /// `ffn_moe_topk-l` ([`topk_ids_logical_within`]), `N_USED` per token,
    /// each in `0..N_EXPERT`.
    fn dump_ids(set: &SetData, l: usize) -> Result<Vec<u32>, GateError> {
        let name = format!("ffn_moe_topk-{l}");
        let row = set.man.tensor(&name, 0)?;
        let ids = topk_ids_logical_within(&set.man, row, u32::try_from(N_EXPERT)?)?;
        if ids.len() != N_USED * set.t {
            return Err(format!(
                "{}: {name} holds {} ids, want {}",
                set.name,
                ids.len(),
                N_USED * set.t
            )
            .into());
        }
        Ok(ids.into_iter().map(i32::cast_unsigned).collect())
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

    /// One output row of a dot, in f64: against our rule's activations and
    /// against ik's, and each dot's sum of absolute terms.
    #[derive(Clone, Copy, Default)]
    struct RowStat {
        ours: f64,
        ik: f64,
        abs_ours: f64,
        abs_ik: f64,
    }

    /// Every row of `w` (`rows` rows of `xo.len()` values) against both
    /// activations. The products of two f32 are exact in f64; the f64 sums
    /// round `k` times at 2^-53, far below the u-scaled bounds they feed.
    fn row_stats(w: &[f32], xo: &[f32], xi: &[f32]) -> Result<Vec<RowStat>, GateError> {
        let k = xo.len();
        let mut out = vec![RowStat::default(); w.len() / k];
        par_chunks(&mut out, 1, |r, o| {
            let mut s = RowStat::default();
            for ((&wv, &a), &b) in w[r * k..(r + 1) * k].iter().zip(xo).zip(xi) {
                let wv = f64::from(wv);
                let (a, b) = (wv * f64::from(a), wv * f64::from(b));
                s.ours += a;
                s.ik += b;
                s.abs_ours += a.abs();
                s.abs_ik += b.abs();
            }
            o[0] = s;
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

    /// `max|a - b| / max|b|` in f64 — the `ik_rel` of the other gates.
    fn rel(a: &[f32], b: &[f32]) -> f64 {
        let den = b.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
        let num = a.iter().zip(b).fold(0.0f64, |m, (&x, &y)| {
            m.max((f64::from(x) - f64::from(y)).abs())
        });
        if den > 0.0 { num / den } else { num }
    }

    /// A dot site's comparisons over one output vector (module doc): (1) the
    /// kernel against our rule's f64 value, `max_rel_err`; (2) the dump
    /// against ik's f64 value, the ratio to ik's accumulation bound; (3) the
    /// kernel against the dump, the ratio to the two rules' band.
    fn dot_cmp(
        kernel: &[f32],
        dump: &[f32],
        st: &[RowStat],
        n_o: f64,
        n_i: f64,
    ) -> Result<(f32, f64, f64), GateError> {
        let ours: Vec<f32> = st.iter().map(|s| s.ours as f32).collect();
        let r1 = max_rel_err(kernel, &ours)?;
        let (mut r2, mut r3) = (0.0f64, 0.0f64);
        for ((&k, &d), s) in kernel.iter().zip(dump).zip(st) {
            let (ro, ri) = (n_o * U * s.abs_ours, n_i * U * s.abs_ik);
            r2 = worst(r2, (f64::from(d) - s.ik).abs(), ri);
            r3 = worst(
                r3,
                (f64::from(k) - f64::from(d)).abs(),
                (s.ours - s.ik).abs() + ro + ri,
            );
        }
        Ok((r1, r2, r3))
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

    /// A SwiGLU row's two bounds (module doc): on `|dump - H(ik)|` and on
    /// `|kernel - dump|`, with ik's centre `H(ik)`.
    fn swiglu_band(g: &RowStat, u: &RowStat, n_o: f64, n_i: f64, limit: f32) -> (f64, f64, f64) {
        let (rgo, rgi) = (n_o * U * g.abs_ours, n_i * U * g.abs_ik);
        let (ruo, rui) = (n_o * U * u.abs_ours, n_i * U * u.abs_ik);
        let cap = if limit > 1e-6 {
            f64::from(limit)
        } else {
            f64::INFINITY
        };
        let lg = SILU_SLOPE * (u.ours.abs().max(u.ik.abs()) + ruo.max(rui)).min(cap);
        let lu = (g.ours.abs().max(g.ik.abs()) + rgo.max(rgi))
            .max(SILU_FLOOR)
            .min(cap);
        let (ho, hi) = (swiglu64(g.ours, u.ours, limit), swiglu64(g.ik, u.ik, limit));
        let (acc_o, acc_i) = (lg * rgo + lu * ruo, lg * rgi + lu * rui);
        let band2 = acc_i + SILU_ROUNDING * (hi.abs() + acc_i);
        let band3 = (ho - hi).abs()
            + acc_o
            + acc_i
            + SILU_ROUNDING * (ho.abs() + acc_o)
            + SILU_ROUNDING * (hi.abs() + acc_i);
        (hi, band2, band3)
    }

    /// A SwiGLU site's comparisons over one output vector: (2) the dump
    /// against ik's f64 SwiGLU, (3) the kernel against the dump — ratios to
    /// [`swiglu_band`]'s bounds.
    fn swiglu_cmp(
        kernel: &[f32],
        dump: &[f32],
        g: &[RowStat],
        u: &[RowStat],
        n_o: f64,
        n_i: f64,
        limit: f32,
    ) -> (f64, f64) {
        let (mut r2, mut r3) = (0.0f64, 0.0f64);
        for (((&k, &d), gs), us) in kernel.iter().zip(dump).zip(g).zip(u) {
            let (hi, b2, b3) = swiglu_band(gs, us, n_o, n_i, limit);
            r2 = worst(r2, (f64::from(d) - hi).abs(), b2);
            r3 = worst(r3, (f64::from(k) - f64::from(d)).abs(), b3);
        }
        (r2, r3)
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
    /// slot order and narrowed, each divided by the sum, then scaled. Returns
    /// (scaled, sum, divided).
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
        let norm = g.map(|v| v / sum);
        (norm.map(|v| v * scale), sum, norm)
    }

    /// The exact weights of the scores at `ids`.
    fn weights64(p: &[f32], ids: &[u32], scale: f32) -> [f64; N_USED] {
        let g: Vec<f64> = ids.iter().map(|&i| f64::from(p[i as usize])).collect();
        let s: f64 = g.iter().sum();
        let mut out = [0.0f64; N_USED];
        for (o, v) in out.iter_mut().zip(&g) {
            *o = v / s * f64::from(scale);
        }
        out
    }

    /// The combine, our rule and ik's: slot 0's product, then fused
    /// multiply-adds in slot order, then `+ shexp`.
    fn combine_host(down: &[f32], w: &[f32], shexp: &[f32]) -> Vec<f32> {
        let rows = shexp.len();
        (0..rows)
            .map(|d| {
                let mut acc = down[d] * w[0];
                for (j, &wj) in w.iter().enumerate().take(N_USED).skip(1) {
                    acc = down[j * rows + d].mul_add(wj, acc);
                }
                acc + shexp[d]
            })
            .collect()
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
        sy: DeviceBuffer<f32>,
        cd: DeviceBuffer<f32>,
        cw: DeviceBuffer<f32>,
        cs: DeviceBuffer<f32>,
        negz: DeviceBuffer<f32>,
        y0: DeviceBuffer<f32>,
        y1: DeviceBuffer<f32>,
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
                sy: z(m.n_embd)?,
                cd: z(N_USED * m.n_embd)?,
                cw: z(N_USED)?,
                cs: z(m.n_embd)?,
                negz: DeviceBuffer::from_host(stream, &vec![-0.0f32; m.n_embd])?,
                y0: z(m.n_embd)?,
                y1: z(m.n_embd)?,
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
    }

    /// One layer's router and shared-expert weights: f32 host copies for the
    /// f64 references and the device copies the kernels read.
    struct Layer {
        l: usize,
        router_w: Vec<f32>,
        router_dev: DeviceTensor<f32>,
        bias: Vec<f32>,
        bias_dev: DeviceBuffer<f32>,
        sh_gate: Vec<f32>,
        sh_up: Vec<f32>,
        sh_down: Vec<f32>,
        sh_gate_dev: DevWeight,
        sh_up_dev: DevWeight,
        sh_down_dev: DevWeight,
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
            let q =
                |name: &str, rows: usize, kk: usize| -> Result<(Vec<f32>, DevWeight), GateError> {
                    let b = tensor(
                        &cx.split,
                        &format!("blk.{l}.{name}.weight"),
                        GgmlType::Q8_0,
                        &[kk as u64, rows as u64],
                    )?;
                    Ok((
                        rows_f32(GgmlType::Q8_0, b, 0, rows, kk)?,
                        q8_0_weight(stream, b, rows, kk)?,
                    ))
                };
            let (sh_gate, sh_gate_dev) = q("ffn_gate_shexp", ff, k)?;
            let (sh_up, sh_up_dev) = q("ffn_up_shexp", ff, k)?;
            let (sh_down, sh_down_dev) = q("ffn_down_shexp", k, ff)?;
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
                sh_gate_dev,
                sh_up_dev,
                sh_down_dev,
                down_ty,
            })
        }
    }

    /// The q8_0 planes of a shared-expert weight, for the op-path gemv.
    fn planes(w: &DevWeight) -> Result<(&DeviceTensor<u32>, &DeviceTensor<u16>), GateError> {
        match w {
            DevWeight::Q8_0 { qs, d, .. } => Ok((qs, d)),
            _ => Err("a shared-expert weight is not q8_0".into()),
        }
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
        let split = Split::open(ref_model_path()?)?;
        let meta = Meta::read(&split)?;
        let gpu = Gpu::new()?;
        let router = RouterKernels::load(gpu.context())?;
        let experts = ExpertKernels::load(gpu.context())?;
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
        };
        let stream = cx.gpu.stream();
        let mut dv = Dev::new(stream, &cx.meta)?;

        let table = oracle::for_arch(Arch::Deepseek41)?;
        let mut sets = Vec::with_capacity(1 + STEP_SETS.len());
        for name in std::iter::once(table.set_name(Set::Cpu)?).chain(STEP_SETS) {
            sets.push(set_data(name, table.open_named(name)?)?);
        }

        let mut tally = Tally::default();
        tally.add(router_ties(&cx, &mut dv)?);

        let mut first_routed = true;
        for l in 0..cx.meta.n_layer {
            let ly = Layer::load(&cx, l)?;
            for set in &sets {
                tally.add(router_site(&cx, &ly, set, &mut dv)?);
                tally.add(shared_site(&cx, &ly, set, &mut dv)?);
                tally.add(combine_site(&cx, set, l, &mut dv)?);
            }
            if ly.down_ty == GgmlType::Q4_K {
                let (pass, extra) = routed_layer(&cx, &ly, &sets, &mut dv, first_routed)?;
                for p in pass {
                    tally.add(p);
                }
                for p in extra {
                    tally.add(p);
                }
                first_routed = false;
            } else {
                println!(
                    "routed layer={l} skipped: ffn_down_exps is {} — the host tier's in every plan",
                    ly.down_ty
                );
            }
        }
        if first_routed {
            return Err("no layer has a q4_K down stack: the routed kernels were not gated".into());
        }

        let pass = tally.failed == 0;
        println!(
            "gate_deepseek41_moe: {} sites across {} sets and {} layers, {} failed — {}",
            tally.sites,
            sets.len(),
            cx.meta.n_layer,
            tally.failed,
            verdict(pass)
        );
        if !pass {
            return Err(checks_failed());
        }
        Ok(())
    }

    /// A set opened through the table (its `# arch` checked there), and its
    /// token count.
    fn set_data(name: &'static str, man: RefManifest) -> Result<SetData, GateError> {
        let t = usize::try_from(man.tensor("ffn_norm-0", 0)?.ne[1])?;
        println!(
            "set {name}: build {:?} tokens {t} complete {:?}",
            man.build, man.complete
        );
        Ok(SetData { name, man, t })
    }

    /// The router at one layer of one set.
    fn router_site(cx: &Cx, ly: &Layer, set: &SetData, dv: &mut Dev) -> Result<bool, GateError> {
        let stream = cx.gpu.stream();
        let (k, t, l, scale) = (cx.meta.n_embd, set.t, ly.l, cx.meta.scale);
        let (k64, t64, ne64, nu64) = (k as u64, t as u64, N_EXPERT as u64, N_USED as u64);
        let norm = format!("ffn_norm-{l}");
        let x = dump(set, &norm, [k64, t64, 1, 1], "in", None)?;
        let logits_d = dump(
            set,
            &format!("ffn_moe_logits-{l}"),
            [ne64, t64, 1, 1],
            "MUL_MAT",
            Some((&format!("blk.{l}.ffn_gate_inp.weight"), &norm)),
        )?;
        let probs_d = dump(
            set,
            &format!("ffn_moe_probs-{l}"),
            [ne64, t64, 1, 1],
            "SQRT_SOFTPLUS",
            Some((&format!("ffn_moe_logits-{l}"), "-")),
        )?;
        let biased_d = dump(
            set,
            &format!("ffn_moe_probs_biased-{l}"),
            [ne64, t64, 1, 1],
            "ADD",
            Some((
                &format!("ffn_moe_probs-{l}"),
                &format!("blk.{l}.exp_probs_b.bias"),
            )),
        )?;
        let ids_d = dump_ids(set, l)?;
        let sum_d = dump(
            set,
            &format!("ffn_moe_weights_sum-{l}"),
            [1, t64, 1, 1],
            "SUM_ROWS",
            None,
        )?;
        let norm_d = dump(
            set,
            &format!("ffn_moe_weights_norm-{l}"),
            [nu64, t64, 1, 1],
            "DIV",
            Some((
                &format!("ffn_moe_weights-{l} (reshaped)"),
                &format!("ffn_moe_weights_sum-{l}"),
            )),
        )?;
        let w_d = dump(
            set,
            &format!("ffn_moe_weights_scaled-{l}"),
            [1, nu64, t64, 1],
            "SCALE",
            None,
        )?;

        let (n_o, n_i) = (n_ours(k), n_ik_f32(k));
        let mut ok1 = true; // logits bits = f32_gemv, ids, weights
        let mut rel1 = 0.0f32; // logits vs f64
        let mut p1 = 0.0f64; // scores vs host, ratio
        let mut ok2 = true; // ik's rule, bit for bit
        let mut r2 = 0.0f64; // dump logits vs f64, ratio
        let mut ids3 = true;
        let (mut l3, mut p3, mut w3) = (0.0f64, 0.0f64, 0.0f64);
        let (mut ik_rel_p, mut ik_rel_w) = (0.0f64, 0.0f64);
        let mut rerun = true;
        let mut tickets_zero = true;
        for tt in 0..t {
            let xt = &x[tt * k..(tt + 1) * k];
            dv.x.copy_from_host(stream, xt)?;
            cx.router.enqueue_router(
                stream,
                &ly.router_dev,
                &dv.x,
                &ly.bias_dev,
                scale,
                &mut dv.rout,
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

            // (1) our rule on the host.
            let st = row_stats(&ly.router_w, xt, xt)?;
            let ours: Vec<f32> = st.iter().map(|s| s.ours as f32).collect();
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

            // (2) ik's rule on the host against the dump.
            let ld = &logits_d[tt * N_EXPERT..(tt + 1) * N_EXPERT];
            let pd = &probs_d[tt * N_EXPERT..(tt + 1) * N_EXPERT];
            let bd = &biased_d[tt * N_EXPERT..(tt + 1) * N_EXPERT];
            let idd = &ids_d[tt * N_USED..(tt + 1) * N_USED];
            let wd = &w_d[tt * N_USED..(tt + 1) * N_USED];
            for (&v, s) in ld.iter().zip(&st) {
                r2 = worst(r2, (f64::from(v) - s.ik).abs(), n_i * U * s.abs_ik);
            }
            let p_sim: Vec<f32> = ld.iter().map(|&v| sqrt_softplus(v)).collect();
            let b_sim: Vec<f32> = pd.iter().zip(&ly.bias).map(|(&p, &b)| p + b).collect();
            let (w_sim, s_sim, n_sim) = weights_rule(pd, idd, scale);
            ok2 &= bits_equal(&p_sim, pd)
                && bits_equal(&b_sim, bd)
                && select(bd)?.as_slice() == idd
                && s_sim.to_bits() == sum_d[tt].to_bits()
                && bits_equal(&n_sim, &norm_d[tt * N_USED..(tt + 1) * N_USED])
                && bits_equal(&w_sim, wd);

            // (3) the kernel against the dump.
            ids3 &= idk.as_slice() == idd;
            for ((&a, &b), s) in lk.iter().zip(ld).zip(&st) {
                l3 = worst(
                    l3,
                    (f64::from(a) - f64::from(b)).abs(),
                    (n_o + n_i) * U * s.abs_ours.max(s.abs_ik),
                );
            }
            for ((&a, &b), (&xa, &xb)) in pk.iter().zip(pd).zip(lk.iter().zip(ld)) {
                let band = (softplus64(xa) - softplus64(xb)).abs()
                    + softplus_err(xa, DEVICE_EXP_ULPS, DEVICE_LOG_ULPS)
                    + softplus_err(xb, HOST_EXP_ULPS, HOST_LOG_ULPS);
                p3 = worst(p3, (f64::from(a) - f64::from(b)).abs(), band);
            }
            ik_rel_p = ik_rel_p.max(rel(&pk, pd));
            if idk.as_slice() == idd {
                let (e_k, e_d) = (weights64(&pk, &idk, scale), weights64(pd, idd, scale));
                for (s, (&a, &b)) in wk.iter().zip(wd).enumerate() {
                    let band =
                        (e_k[s] - e_d[s]).abs() + WEIGHT_ROUNDING * (e_k[s].abs() + e_d[s].abs());
                    w3 = worst(w3, (f64::from(a) - f64::from(b)).abs(), band);
                }
                ik_rel_w = ik_rel_w.max(rel(&wk, wd));
            }
        }
        let pass = ok1
            && rel1 <= KERNEL_BAND
            && p1 <= 1.0
            && ok2
            && r2 <= 1.0
            && ids3
            && l3 <= 1.0
            && p3 <= 1.0
            && w3 <= 1.0
            && rerun
            && tickets_zero;
        println!(
            "router set={} layer={l} tokens={t} logits_eq_f32_gemv+ids+weights_exact={ok1} logits_rel={rel1:.3e} \
             probs_ratio={p1:.3} | ik_rule_bits={ok2} ik_logits_ratio={r2:.3e} | ids_eq_dump={ids3} \
             logits_ratio={l3:.3e} probs_ratio={p3:.3} weights_ratio={w3:.3} probs_ik_rel={ik_rel_p:.3e} \
             weights_ik_rel={ik_rel_w:.3e} rerun={rerun} tickets_zero={tickets_zero} {}",
            set.name,
            verdict(pass)
        );
        Ok(pass)
    }

    /// The shared expert at one layer of one set.
    fn shared_site(cx: &Cx, ly: &Layer, set: &SetData, dv: &mut Dev) -> Result<bool, GateError> {
        let stream = cx.gpu.stream();
        let (k, ff, t, l) = (cx.meta.n_embd, cx.meta.n_ff, set.t, ly.l);
        let limit = cx.meta.clamp_shexp[l];
        let (k64, ff64, t64) = (k as u64, ff as u64, t as u64);
        let x = dump(set, &format!("ffn_norm-{l}"), [k64, t64, 1, 1], "in", None)?;
        let upg = format!("ffn_up_gate-{l}");
        let h_d = dump(
            set,
            &upg,
            [ff64, t64, 1, 1],
            "FUSED_UP_GATE",
            Some((
                &format!("blk.{l}.ffn_up_shexp.weight"),
                &format!("blk.{l}.ffn_gate_shexp.weight"),
            )),
        )?;
        let y_d = dump(
            set,
            &format!("ffn_shexp-{l}"),
            [k64, t64, 1, 1],
            "MUL_MAT",
            Some((&format!("blk.{l}.ffn_down_shexp.weight"), &upg)),
        )?;
        let (gq, gd) = planes(&ly.sh_gate_dev)?;
        let (uq, ud) = planes(&ly.sh_up_dev)?;
        let (dq, dd) = planes(&ly.sh_down_dev)?;
        let (no_k, ni_k, no_f, ni_f) = (n_ours(k), n_ik_quant(k), n_ours(ff), n_ik_quant(ff));
        let mut h_exact = true;
        let (mut g_rel, mut u_rel, mut y_rel) = (0.0f32, 0.0f32, 0.0f32);
        let (mut h2, mut h3, mut y2, mut y3) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (mut h_ik, mut y_ik) = (0.0f64, 0.0f64);
        let mut rerun = true;
        let mut hits = [0usize; 3];
        for tt in 0..t {
            let xt = &x[tt * k..(tt + 1) * k];
            let ht = &h_d[tt * ff..(tt + 1) * ff];
            let yt = &y_d[tt * k..(tt + 1) * k];
            dv.x.copy_from_host(stream, xt)?;
            let q8 = cx.gpu.q8f32();
            q8.enqueue_q8_0_gemv(stream, gq, gd, &dv.x, 1, &mut dv.sg)?;
            q8.enqueue_q8_0_gemv(stream, uq, ud, &dv.x, 1, &mut dv.su)?;
            cx.experts.enqueue_shexp_gate_up(
                stream,
                &ly.sh_gate_dev,
                &ly.sh_up_dev,
                &dv.x,
                limit,
                &mut dv.sh,
            )?;
            let (g_op, u_op, hk) = (
                dv.sg.to_host_vec(stream)?,
                dv.su.to_host_vec(stream)?,
                dv.sh.to_host_vec(stream)?,
            );
            cx.experts.enqueue_shexp_gate_up(
                stream,
                &ly.sh_gate_dev,
                &ly.sh_up_dev,
                &dv.x,
                limit,
                &mut dv.sh,
            )?;
            rerun &= bits_equal(&hk, &dv.sh.to_host_vec(stream)?);
            dv.shin.copy_from_host(stream, ht)?;
            q8.enqueue_q8_0_gemv(stream, dq, dd, &dv.shin, 1, &mut dv.sy)?;
            let yk = dv.sy.to_host_vec(stream)?;
            q8.enqueue_q8_0_gemv(stream, dq, dd, &dv.shin, 1, &mut dv.sy)?;
            rerun &= bits_equal(&yk, &dv.sy.to_host_vec(stream)?);

            let h_host: Vec<f32> = g_op
                .iter()
                .zip(&u_op)
                .map(|(&g, &u)| swiglu_clamp(g, u, limit))
                .collect();
            h_exact &= bits_equal(&hk, &h_host);
            let (cs, cu, cl) = crossings(&g_op, &u_op, limit);
            hits = [hits[0] + cs, hits[1] + cu, hits[2] + cl];
            let mut xi = vec![0.0f32; k];
            quantize_row_q8_2_x4_roundtrip(xt, &mut xi);
            let gs = row_stats(&ly.sh_gate, xt, &xi)?;
            let us = row_stats(&ly.sh_up, xt, &xi)?;
            let g64: Vec<f32> = gs.iter().map(|s| s.ours as f32).collect();
            let u64_: Vec<f32> = us.iter().map(|s| s.ours as f32).collect();
            g_rel = g_rel.max(max_rel_err(&g_op, &g64)?);
            u_rel = u_rel.max(max_rel_err(&u_op, &u64_)?);
            let (a, b) = swiglu_cmp(&hk, ht, &gs, &us, no_k, ni_k, limit);
            h2 = h2.max(a);
            h3 = h3.max(b);
            h_ik = h_ik.max(rel(&hk, ht));

            let mut hi = vec![0.0f32; ff];
            quantize_row_q8_2_x4_roundtrip(ht, &mut hi);
            let ds = row_stats(&ly.sh_down, ht, &hi)?;
            let (a, b, c) = dot_cmp(&yk, yt, &ds, no_f, ni_f)?;
            y_rel = y_rel.max(a);
            y2 = y2.max(b);
            y3 = y3.max(c);
            y_ik = y_ik.max(rel(&yk, yt));
        }
        let pass = h_exact
            && g_rel <= KERNEL_BAND
            && u_rel <= KERNEL_BAND
            && y_rel <= KERNEL_BAND
            && h2 <= 1.0
            && h3 <= 1.0
            && y2 <= 1.0
            && y3 <= 1.0
            && rerun;
        println!(
            "shared set={} layer={l} tokens={t} limit={limit} clamp_hits={}/{}/{} h_eq_host={h_exact} \
             gate_rel={g_rel:.3e} up_rel={u_rel:.3e} down_rel={y_rel:.3e} | ik_h_ratio={h2:.3} \
             ik_down_ratio={y2:.3} | h_ratio={h3:.3} h_ik_rel={h_ik:.3e} down_ratio={y3:.3} \
             down_ik_rel={y_ik:.3e} rerun={rerun} {}",
            set.name,
            hits[0],
            hits[1],
            hits[2],
            verdict(pass)
        );
        Ok(pass)
    }

    /// The combine at one layer of one set, from the dump's inputs.
    fn combine_site(cx: &Cx, set: &SetData, l: usize, dv: &mut Dev) -> Result<bool, GateError> {
        let stream = cx.gpu.stream();
        let (k, t) = (cx.meta.n_embd, set.t);
        let (k64, t64, nu64) = (k as u64, t as u64, N_USED as u64);
        let (moe, sh, down, ws) = (
            format!("ffn_moe_out-{l}"),
            format!("ffn_shexp-{l}"),
            format!("ffn_moe_down-{l}"),
            format!("ffn_moe_weights_scaled-{l}"),
        );
        let down_d = dump(set, &down, [k64, nu64, t64, 1], "MUL_MAT_ID", None)?;
        let w_d = dump(set, &ws, [1, nu64, t64, 1], "SCALE", None)?;
        let moe_d = dump(
            set,
            &moe,
            [k64, t64, 1, 1],
            "MUL_MULTI_ADD",
            Some((&down, &ws)),
        )?;
        let sh_d = dump(set, &sh, [k64, t64, 1, 1], "MUL_MAT", None)?;
        let out_d = dump(
            set,
            &format!("ffn_out-{l}"),
            [k64, t64, 1, 1],
            "ADD",
            Some((&moe, &sh)),
        )?;
        let negz = vec![-0.0f32; k];
        let (mut ok1, mut ok2, mut ok3, mut rerun) = (true, true, true, true);
        for tt in 0..t {
            let dt = &down_d[tt * N_USED * k..(tt + 1) * N_USED * k];
            let wt = &w_d[tt * N_USED..(tt + 1) * N_USED];
            let st = &sh_d[tt * k..(tt + 1) * k];
            let mt = &moe_d[tt * k..(tt + 1) * k];
            let ot = &out_d[tt * k..(tt + 1) * k];
            dv.cd.copy_from_host(stream, dt)?;
            dv.cw.copy_from_host(stream, wt)?;
            dv.cs.copy_from_host(stream, st)?;
            cx.experts
                .enqueue_moe_combine(stream, &dv.cd, &dv.cw, &dv.negz, k, &mut dv.y0)?;
            cx.experts
                .enqueue_moe_combine(stream, &dv.cd, &dv.cw, &dv.cs, k, &mut dv.y1)?;
            let (y0, y1) = (dv.y0.to_host_vec(stream)?, dv.y1.to_host_vec(stream)?);
            cx.experts
                .enqueue_moe_combine(stream, &dv.cd, &dv.cw, &dv.cs, k, &mut dv.y1)?;
            rerun &= bits_equal(&y1, &dv.y1.to_host_vec(stream)?);
            let (h0, h1) = (combine_host(dt, wt, &negz), combine_host(dt, wt, st));
            ok1 &= bits_equal(&y0, &h0) && bits_equal(&y1, &h1);
            let sum: Vec<f32> = mt.iter().zip(st).map(|(&a, &b)| a + b).collect();
            ok2 &= bits_equal(&h0, mt) && bits_equal(&sum, ot);
            ok3 &= bits_equal(&y0, mt) && bits_equal(&y1, ot);
        }
        let pass = ok1 && ok2 && ok3 && rerun;
        println!(
            "combine set={} layer={l} tokens={t} kernel_eq_host={ok1} | ik_rule_bits={ok2} | \
             moe_out_and_ffn_out_eq_dump={ok3} rerun={rerun} {}",
            set.name,
            verdict(pass)
        );
        Ok(pass)
    }

    /// One set's routed inputs at a layer: per token the activations, our
    /// and ik's quantizations of them, the dump's ids, h and down.
    struct RoutedIn {
        x: Vec<f32>,
        xo: Vec<f32>,
        xi: Vec<f32>,
        ids: Vec<u32>,
        h: Vec<f32>,
        down: Vec<f32>,
    }

    /// Set `set`'s routed inputs at layer `l`, each row's chain proven.
    fn routed_in(cx: &Cx, set: &SetData, l: usize) -> Result<RoutedIn, GateError> {
        let (k, ff) = (cx.meta.n_embd, cx.meta.n_ff);
        let (k64, ff64, nu64, t64) = (k as u64, ff as u64, N_USED as u64, set.t as u64);
        let x = dump(set, &format!("ffn_norm-{l}"), [k64, t64, 1, 1], "in", None)?;
        let par = format!("ffn_moe_gate_par-{l}");
        let h = dump(
            set,
            &par,
            [ff64, nu64, t64, 1],
            "MOE_FUSED_UP_GATE",
            Some((
                &format!("blk.{l}.ffn_up_exps.weight"),
                &format!("blk.{l}.ffn_gate_exps.weight"),
            )),
        )?;
        let down = dump(
            set,
            &format!("ffn_moe_down-{l}"),
            [k64, nu64, t64, 1],
            "MUL_MAT_ID",
            Some((&format!("blk.{l}.ffn_down_exps.weight"), &par)),
        )?;
        let xo = q8_1_dequant(&x, k, set.t);
        let mut xi = vec![0.0f32; x.len()];
        for (a, b) in x.chunks_exact(k).zip(xi.chunks_exact_mut(k)) {
            quantize_activations(GgmlType::Q3_K, a, b)?;
        }
        Ok(RoutedIn {
            x,
            xo,
            xi,
            ids: dump_ids(set, l)?,
            h,
            down,
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

    /// A layer's routed stacks on the card: the experts any set's tokens use,
    /// in id order.
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

    /// The per-row statistics of one (token, slot): gate, up, down.
    #[derive(Default, Clone)]
    struct SlotStats {
        g: Vec<RowStat>,
        u: Vec<RowStat>,
        d: Vec<RowStat>,
    }

    /// Every (set, token, slot)'s row statistics, expert by expert — each
    /// expert's weights decoded once.
    fn routed_stats(
        cx: &Cx,
        file: &RoutedFile<'_>,
        ins: &[RoutedIn],
        experts: &[u32],
    ) -> Result<Vec<Vec<SlotStats>>, GateError> {
        let (k, ff) = (cx.meta.n_embd, cx.meta.n_ff);
        let mut stats: Vec<Vec<SlotStats>> = ins
            .iter()
            .map(|i| vec![SlotStats::default(); i.ids.len()])
            .collect();
        for &e in experts {
            let eu = e as usize;
            let wg = rows_f32(GgmlType::Q3_K, file.gate, eu * ff, ff, k)?;
            let wu = rows_f32(GgmlType::Q3_K, file.up, eu * ff, ff, k)?;
            let wd = rows_f32(GgmlType::Q4_K, file.down, eu * k, k, ff)?;
            for (inp, st) in ins.iter().zip(stats.iter_mut()) {
                for (j, _) in inp.ids.iter().enumerate().filter(|&(_, &id)| id == e) {
                    let tt = j / N_USED;
                    let (xo, xi) = (&inp.xo[tt * k..(tt + 1) * k], &inp.xi[tt * k..(tt + 1) * k]);
                    let hs = &inp.h[j * ff..(j + 1) * ff];
                    let ho = q8_1_dequant(hs, ff, 1);
                    let mut hi = vec![0.0f32; ff];
                    quantize_activations(GgmlType::Q4_K, hs, &mut hi)?;
                    st[j] = SlotStats {
                        g: row_stats(&wg, xo, xi)?,
                        u: row_stats(&wu, xo, xi)?,
                        d: row_stats(&wd, &ho, &hi)?,
                    };
                }
            }
        }
        Ok(stats)
    }

    /// The routed experts at one layer, every set: the stacks of the experts
    /// any set's tokens use, compacted and uploaded once. Returns the sites'
    /// verdicts and, at the first routed layer, the synthetic checks'.
    fn routed_layer(
        cx: &Cx,
        ly: &Layer,
        sets: &[SetData],
        dv: &mut Dev,
        synthetic: bool,
    ) -> Result<(Vec<bool>, Vec<bool>), GateError> {
        let l = ly.l;
        let ins: Vec<RoutedIn> = sets
            .iter()
            .map(|set| routed_in(cx, set, l))
            .collect::<Result<_, _>>()?;
        let mut experts: Vec<u32> = ins.iter().flat_map(|i| i.ids.iter().copied()).collect();
        experts.sort_unstable();
        experts.dedup();
        let file = RoutedFile::open(cx, l)?;
        let stats = routed_stats(cx, &file, &ins, &experts)?;
        let stacks = file.upload(cx, experts)?;
        let mut verdicts = Vec::new();
        for ((set, inp), st) in sets.iter().zip(&ins).zip(&stats) {
            verdicts.push(routed_set(cx, ly, set, inp, st, &stacks, dv)?);
        }
        let mut extra = Vec::new();
        if synthetic {
            let k = cx.meta.n_embd;
            let (x, sel) = (&ins[0].x[..k], stacks.sel(&ins[0].ids, 0));
            extra.push(clamp_routed(cx, dv, &stacks, &sel, x, l)?);
            extra.push(clamp_shared(cx, dv, ly, x)?);
            extra.push(out_of_range(
                cx,
                dv,
                &stacks,
                &sel,
                x,
                cx.meta.clamp_exp[l],
            )?);
            extra.push(graph(cx, dv, ly, &stacks, &sel, x)?);
        }
        Ok((verdicts, extra))
    }

    /// The routed site of one set at one layer: per token the op-path gemvs,
    /// the kernel and its rerun, the down from the dump's h, and the three
    /// comparisons.
    fn routed_set(
        cx: &Cx,
        ly: &Layer,
        set: &SetData,
        inp: &RoutedIn,
        st: &[SlotStats],
        stacks: &Stacks,
        dv: &mut Dev,
    ) -> Result<bool, GateError> {
        let stream = cx.gpu.stream();
        let (k, ff, l) = (cx.meta.n_embd, cx.meta.n_ff, ly.l);
        let limit = cx.meta.clamp_exp[l];
        let (no_k, ni_k) = (n_ours(k), n_ik_quant(k));
        let (no_f, ni_f) = (n_ours(ff) + Q4K_DECODE, n_ik_quant(ff) + Q4K_DECODE);
        let mut h_exact = true;
        let (mut g_rel, mut u_rel, mut d_rel) = (0.0f32, 0.0f32, 0.0f32);
        let (mut h2, mut h3, mut d2, mut d3) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (mut h_ik, mut d_ik) = (0.0f64, 0.0f64);
        let mut rerun = true;
        let mut hits = [0usize; 3];
        for tt in 0..set.t {
            let sel_dev = DeviceBuffer::from_host(stream, &stacks.sel(&inp.ids, tt))?;
            dv.x.copy_from_host(stream, &inp.x[tt * k..(tt + 1) * k])?;
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
            };
            cx.experts
                .enqueue_expert_gate_up(stream, &args, &mut dv.h6)?;
            let (g_op, u_op) = (dv.g6.to_host_vec(stream)?, dv.u6.to_host_vec(stream)?);
            let hk = dv.h6.to_host_vec(stream)?;
            cx.experts
                .enqueue_expert_gate_up(stream, &args, &mut dv.h6)?;
            rerun &= bits_equal(&hk, &dv.h6.to_host_vec(stream)?);

            let h_d = &inp.h[tt * N_USED * ff..(tt + 1) * N_USED * ff];
            dv.hin.copy_from_host(stream, h_d)?;
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
            let d_d = &inp.down[tt * N_USED * k..(tt + 1) * N_USED * k];
            for (s, ss) in st[tt * N_USED..(tt + 1) * N_USED].iter().enumerate() {
                let (a, b) = (s * ff, (s + 1) * ff);
                let g64: Vec<f32> = ss.g.iter().map(|v| v.ours as f32).collect();
                let u64_: Vec<f32> = ss.u.iter().map(|v| v.ours as f32).collect();
                g_rel = g_rel.max(max_rel_err(&g_op[a..b], &g64)?);
                u_rel = u_rel.max(max_rel_err(&u_op[a..b], &u64_)?);
                let (r2, r3) = swiglu_cmp(&hk[a..b], &h_d[a..b], &ss.g, &ss.u, no_k, ni_k, limit);
                h2 = h2.max(r2);
                h3 = h3.max(r3);
                let (a, b) = (s * k, (s + 1) * k);
                let (r1, r2, r3) = dot_cmp(&dk[a..b], &d_d[a..b], &ss.d, no_f, ni_f)?;
                d_rel = d_rel.max(r1);
                d2 = d2.max(r2);
                d3 = d3.max(r3);
            }
            h_ik = h_ik.max(rel(&hk, h_d));
            d_ik = d_ik.max(rel(&dk, d_d));
        }
        let pass = h_exact
            && g_rel <= KERNEL_BAND
            && u_rel <= KERNEL_BAND
            && d_rel <= KERNEL_BAND
            && h2 <= 1.0
            && h3 <= 1.0
            && d2 <= 1.0
            && d3 <= 1.0
            && rerun;
        println!(
            "routed set={} layer={l} tokens={} experts={} limit={limit} clamp_hits={}/{}/{} \
             h_eq_host={h_exact} gate_rel={g_rel:.3e} up_rel={u_rel:.3e} down_rel={d_rel:.3e} | \
             ik_h_ratio={h2:.3} ik_down_ratio={d2:.3} | h_ratio={h3:.3} h_ik_rel={h_ik:.3e} \
             down_ratio={d3:.3} down_ik_rel={d_ik:.3e} rerun={rerun} {}",
            set.name,
            set.t,
            stacks.experts.len(),
            hits[0],
            hits[1],
            hits[2],
            verdict(pass)
        );
        Ok(pass)
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
        let (gq, gd) = planes(&ly.sh_gate_dev)?;
        let (uq, ud) = planes(&ly.sh_up_dev)?;
        let run = |dv: &mut Dev, x: &[f32]| -> Result<(Vec<f32>, Vec<f32>), GateError> {
            dv.x.copy_from_host(stream, x)?;
            cx.gpu
                .q8f32()
                .enqueue_q8_0_gemv(stream, gq, gd, &dv.x, 1, &mut dv.sg)?;
            cx.gpu
                .q8f32()
                .enqueue_q8_0_gemv(stream, uq, ud, &dv.x, 1, &mut dv.su)?;
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
            cx.experts.enqueue_shexp_gate_up(
                stream,
                &ly.sh_gate_dev,
                &ly.sh_up_dev,
                &dv.x,
                lim,
                &mut dv.sh,
            )?;
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

    /// Enqueue one layer's eight launches for the token in `dv.x`: the router,
    /// the q8_1 of `x`, the routed gate·up·SwiGLU, the q8_1 of `h`, the
    /// routed down, the shared gate·up·SwiGLU, the shared down, the combine
    /// (on the router's weights). The routed slots read `sel` (the dump's
    /// ids in the compact stack), not the router's ids: the gate's stacks
    /// hold only the experts the sets use.
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
        };
        cx.experts
            .enqueue_expert_gate_up(stream, &args, &mut dv.h6)?;
        cx.gpu.enqueue_quantize_q8_1(&dv.h6, &mut dv.act_h)?;
        cx.gpu
            .q4k_sel()
            .enqueue_gemv_q4k_sel(stream, &s.wd, &dv.act_h, sel, N_USED, k, &mut dv.d6)?;
        cx.experts.enqueue_shexp_gate_up(
            stream,
            &ly.sh_gate_dev,
            &ly.sh_up_dev,
            &dv.x,
            cx.meta.clamp_shexp[ly.l],
            &mut dv.sh,
        )?;
        let DevWeight::Q8_0 { qs, d, .. } = &ly.sh_down_dev else {
            return Err(bloomery_gpu::GpuError::Shape {
                what: "layer_chain",
                detail: "the shared down weight is not q8_0".into(),
            });
        };
        cx.gpu
            .q8f32()
            .enqueue_q8_0_gemv(stream, qs, d, &dv.sh, 1, &mut dv.sy)?;
        cx.experts
            .enqueue_moe_combine(stream, &dv.d6, &dv.rout.weights, &dv.sy, k, &mut dv.y1)?;
        Ok(())
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
            dv.y1.to_host_vec(stream)?,
        ])
    }

    /// One layer's eight launches captured and replayed twice: eight graph
    /// nodes, each replay's outputs bit-identical to the eager run's, and the
    /// router's ticket count back at zero after each.
    fn graph(
        cx: &Cx,
        dv: &mut Dev,
        ly: &Layer,
        s: &Stacks,
        sel: &[u32],
        x: &[f32],
    ) -> Result<bool, GateError> {
        const NODES: usize = 8;
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
        let pass = nodes == NODES && same && tickets_zero;
        println!(
            "graph layer={} nodes={nodes} (want {NODES}) replays_eq_eager={same} tickets_zero={tickets_zero} {}",
            ly.l,
            verdict(pass)
        );
        Ok(pass)
    }

    /// The router's selection on planted ties (a K = 32 router whose logits
    /// are its first weight column against a one-hot input), against the
    /// rule's statement and the host's selection from the kernel's scores.
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
            cx.router
                .enqueue_router(stream, &w_dev, &x_dev, &b_dev, scale, &mut dv.rout)?;
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
            let tickets = dv.rout.tickets(stream)?;
            let pass = ids_k.as_slice() == stated.as_slice()
                && ids_k.as_slice() == host.as_slice()
                && bits_equal(&wk, &w_host)
                && logits_exact
                && tickets == 0;
            println!(
                "ties case={name} ids={ids_k:?} stated={stated:?} host={host:?} weights_exact={} logits_exact={logits_exact} \
                 tickets={tickets} {}",
                bits_equal(&wk, &w_host),
                verdict(pass)
            );
            all &= pass;
        }
        Ok(all)
    }
}
