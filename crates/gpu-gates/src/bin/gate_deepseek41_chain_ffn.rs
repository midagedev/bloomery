//! GPU gate for DeepSeek-V4.1's MoE sub-layer as one chain piece
//! (`bloomery_gpu_deepseek41::chain::ffn::FfnPiece`, B5 G1) against ik's CPU
//! dumps: the decode-step sets at positions 4 and 301 (`STEP4`, `D1N`), T = 1,
//! every layer.
//!
//! Per layer the gate makes the layer's dense tensors resident with
//! `Weights::load_where` and uploads the card's routed stacks compacted in slot
//! order; the host tier is served in-process — a `Hybrid` over the boundary
//! with the V4.1 host computation (`Ds41Host`, the file's `HostLayer`s). Per set
//! it injects the streams the sub-layer reads (`hc_attn_post-L`) and its folded
//! input (`hc_ffn_pre-L`), runs the piece eagerly (the host serves the layer as
//! the piece hands it over) and checks three things:
//!
//! 1. **The piece is the op path, bit for bit.** The gate runs the same op
//!    kernels itself from its own buffers — the norm, the router, the card's
//!    routed experts on places it translates on the host from the slot map, the
//!    shared expert, HC_PRE — computes the host share with its own `HostLayer`
//!    call on its own activation, and the combine ([`combine_elem`]), HC_POST
//!    and the fold on the host. Every buffer the piece leaves must equal it:
//!    the activation in the handoff, the routing, the places, the card's slots
//!    of h and down, the shared expert, the host's partial sum in the mapped
//!    page, the combine, HC_PRE, the streams and the fold. This is the
//!    composition's contract — the op kernels' arithmetic is their own gates'.
//! 2. **ik's composition** — the rows the gate reads are proven by what each
//!    reads (the norm the fold, HC_POST `ffn_out` and the ffn HC_PRE, the next
//!    fold `l_out` or an engram layer's streams), and the piece's form (HC_POST
//!    with the fold or alone) must be the dump's. A broken relation is an error.
//! 3. **The piece against the dump**:
//!    - the routed ids exactly, unless a pair of experts sits within the
//!      selection band of each other where the order is decided (a near tie:
//!      the ids and every pin below are waived for that layer and set, and the
//!      line says so). The band on an expert's selection value is the exact
//!      scores' difference at the two engines' logits, each side's score
//!      evaluation (`softplus_err`: the device's `expf`/`logf`, the host's
//!      libm) and the bias add's half ulp a side. That band is read from the
//!      logits, so a waiver also needs the router's input within the two
//!      norms' rounding of the dump's (`norm_band`): a wrong input moves the
//!      logits, and the band would waive what it caused;
//!    - the combine against `ffn_out-L`, the new streams against `l_out-L` and
//!      the fold against the next sub-layer's (`hc_attn_pre-(L+1)`, when it
//!      reads `l_out-L`), each in the form of the HC gate's chain pin: the
//!      difference `pred` of the two engines' results in exact f64 from each
//!      side's own inputs, and a `bound` on both sides' float rounding around
//!      it; `|(ours − ik) − pred| / bound` above 1 fails. The bounds are the
//!      op bands' rounding terms the B4 MoE gate pinned, carried down the
//!      chain:
//!    - the combine: per slot `pred += w°·D°₆₄ − wᵏ·Dᵏ₆₄`, the down's f64 dot
//!      at each side's activation rule on its own h (q8_1 per 128 of the
//!      card's h, q8_2_x4 of the host's h, q8_2_x4 of the dump's h), and
//!      `bound += |w°|·n°·u·A° + |wᵏ|·nᵏ·u·Aᵏ` (each side's accumulation bound,
//!      the K-quant decode's rounding, and the gate's f32 image of our q8_1
//!      codes); the shared expert's the same way (our f32 h against ik's
//!      q8_2_x4); then the three sums' roundings — ours
//!      `(Σ_card fma) + hsum + shexp`, the host's weighted sum, ik's six-slot
//!      sum plus the shared expert — which is where the association change
//!      the partial sum brings lives;
//!    - the streams: HC_POST in f64 at our combine as the combine's pin has it
//!      (`yᵏ + pred_y`) with our HC_PRE's post and comb, minus ik's at `yᵏ`
//!      with the dump's; `bound = |post°|·bound_y` plus both sides' HC_POST
//!      roundings;
//!    - the fold: the same over the streams' pin with each side's pre.
//!
//!    Every product of two f32 is exact in f64; the gate's own f64 sums add
//!    at most `k·u64` of the magnitudes they sum, kept in each bound.
//!
//! The slot map is synthetic, not the gate placement's prefix: the 3090's prefix
//! of a few dozen of 384 experts would put one routed id of a token on the card
//! about one layer in six, so most layers would run only one side of the join.
//! Here each layer whose down stack is q4_K puts on the card half of the
//! experts the two sets' tokens use — every token has ids on both sides — in
//! slots numbered against the id order, so the translation is not the
//! identity. Layers whose down stack has no device format (q5_K) are all host,
//! as in every plan.
//!
//! Structure, per layer: the piece's launches captured and replayed for each
//! set with the host serving each replay, every buffer bit-identical to the
//! eager run; the captured nodes by kind (kernels, copies — none: the handoff
//! launch writes the page —, the two memory-operation batches) equal to the
//! piece's own count. At the first
//! layer with card experts and a fold, the overlap lever off (the wait right
//! after the go) must leave every buffer as it is.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_deepseek41_chain_ffn: built without the `deepseek41` feature; see `just \
         gate-gpu-ds41-chain-ffn`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_deepseek41_chain_ffn", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use bloomery_gpu::fused::FusedKernels;
    use bloomery_gpu::hybrid::{Boundary, BoundaryShape, HOST, Hybrid, SlotMap};
    use bloomery_gpu::weights::{DevWeight, Weights};
    use bloomery_gpu::{DeviceTensor, Gpu, NodeInfo, Q8Act};
    use bloomery_gpu_deepseek41::chain::ffn::{
        CardStacks, Ds41Host, FfnIo, FfnPiece, combine_elem,
    };
    use bloomery_gpu_deepseek41::experts::{ExpertGateUp, ExpertKernels};
    use bloomery_gpu_deepseek41::hc::{
        HC_MIX, HC_STREAMS, HcKernels, HcParams, HcPreArgs, HcPreScratch,
    };
    use bloomery_gpu_deepseek41::router::{N_EXPERT, N_USED, RouterKernels, RouterOut};
    use bloomery_gpu_gates::oracle;
    use bloomery_gpu_gates::oracle::deepseek41::{D1N, STEP4};
    use bloomery_gpu_gates::{
        GateError, RefManifest, RefRow, bits_equal, bytes_to_words, checks_failed, no_local_depot,
        q8_1_dequant, ref_model_path, ref_tensor_of_in, row_bytes, topk_ids_logical_within,
        verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer, sys};
    use gguf::Split;
    use gguf::quant::{
        GgmlType, dequant_row, quantize_activations, quantize_row_q8_2_x4_roundtrip,
    };
    use model::Tensor2;
    use model::arch::Arch;
    use model::arch::deepseek41::hparams::Hparams;
    use model::arch::deepseek41::{host, names};
    use model::moe::{HostLayer, HostScratch};

    /// The sets, T = 1 each: a decode step at position 4, and one at 301 with
    /// the file's top-k.
    const SETS: [&str; 2] = [STEP4, D1N];

    /// The ik tree the sets must come from: the dump before its sink fix is
    /// kept read-only elsewhere and must not be read.
    const PRE_FIX_BUILD: &str = "49ef19d0";

    /// f32's unit roundoff, 2^-24.
    const U: f64 = f32::EPSILON as f64 / 2.0;

    /// `n·u / (1 - n·u)`: the relative bound of `n` roundings.
    fn gamma(n: usize) -> f64 {
        let nu = n as f64 * U;
        nu / (1.0 - nu)
    }

    /// Roundings on one term's path through one of our dots over `k` values
    /// (the MoE gate's count): one per value a lane reads, `k/32` a lane, and
    /// the five butterfly levels.
    fn n_ours(k: usize) -> f64 {
        (k / 32 + 5) as f64
    }

    /// The same count for ik's and the host's quantized dots, whose order the
    /// gate does not transcribe (the MoE gate's): four roundings per 32-value
    /// block at most, then a horizontal sum of at most five levels.
    fn n_ik_quant(k: usize) -> f64 {
        (4 * (k / 32) + 5) as f64
    }

    /// A K-quant down weight decoded once to f32 against both engines' exact
    /// use of it (the MoE gate's `Q4K_DECODE`): one rounding a side.
    const KQ_DECODE: f64 = 1.0;

    /// The gate's f32 image of our q8_1 activation (`q8_1_dequant`, code
    /// times scale) against the kernel's exact use of both: one rounding a
    /// value.
    const Q8_IMAGE: f64 = 1.0;

    /// f64's unit roundoff, 2^-53: the gate's own sums round too, and each
    /// bound keeps what they can add.
    const U64: f64 = f64::EPSILON / 2.0;

    /// CUDA's `expf` (fused with the `1 +`) and `logf`, the host's (glibc's),
    /// in ulps — the MoE gate's constants.
    const DEVICE_EXP_ULPS: f64 = 2.5;
    const DEVICE_LOG_ULPS: f64 = 1.0;
    const HOST_EXP_ULPS: f64 = 1.0;
    const HOST_LOG_ULPS: f64 = 1.0;

    // ---------------------------------------------------------- the dump

    /// A set, its manifest read once.
    struct SetData {
        name: &'static str,
        man: RefManifest,
    }

    fn set_data(name: &'static str) -> Result<SetData, GateError> {
        let man = oracle::for_arch(Arch::Deepseek41)?.open_named(name)?;
        let t = man.tensor("ffn_norm-0", 0)?.ne[1];
        println!(
            "set {name}: build {:?} tokens {t} complete {:?}",
            man.build, man.complete
        );
        if man.build.as_deref() == Some(PRE_FIX_BUILD) {
            return Err(format!(
                "{name} was dumped by ik {PRE_FIX_BUILD}, the tree before its sink fix: \
                 the sets must be the re-dump"
            )
            .into());
        }
        if t != 1 {
            return Err(format!("{name} has {t} tokens; the piece runs one").into());
        }
        Ok(SetData { name, man })
    }

    fn src0(r: &RefRow) -> &str {
        r.src0.as_deref().unwrap_or("")
    }

    /// Row `name`/0 of `set`, its type, dims and op proven, and its two
    /// sources when given (`None` for a source not checked).
    fn row<'a>(
        set: &'a SetData,
        name: &str,
        ne: [u64; 4],
        op: &str,
        srcs: (Option<&str>, Option<&str>),
    ) -> Result<&'a RefRow, GateError> {
        let r = set.man.tensor(name, 0)?;
        r.expect(name, "f32", ne, op)?;
        for (want, got) in [(srcs.0, &r.src0), (srcs.1, &r.src1)] {
            if let Some(w) = want
                && got.as_deref() != Some(w)
            {
                return Err(format!(
                    "{}: {name} reads {:?} and {:?}; want {:?} and {:?}",
                    set.name, r.src0, r.src1, srcs.0, srcs.1
                )
                .into());
            }
        }
        Ok(r)
    }

    fn values(set: &SetData, r: &RefRow) -> Result<Vec<f32>, GateError> {
        ref_tensor_of_in(&set.man.dir, r)
    }

    /// What one layer of one set holds for the piece: its inputs, ik's
    /// intermediates the bands read, and ik's outputs.
    struct LayerDump {
        streams: Vec<f32>,
        fold_in: Vec<f32>,
        /// The norm's output: the router's input.
        x: Vec<f32>,
        logits: Vec<f32>,
        probs_biased: Vec<f32>,
        ids: [u32; N_USED],
        w: [f32; N_USED],
        h: Vec<f32>,
        down: Vec<f32>,
        sh_h: Vec<f32>,
        sh: Vec<f32>,
        y: Vec<f32>,
        hc: [f32; HC_MIX],
        l_out: Vec<f32>,
        /// The next sub-layer's fold when it reads `l_out-L`; `None` into an
        /// engram layer and after the last layer.
        fold_out: Option<Vec<f32>>,
    }

    fn layer_dump(set: &SetData, hp: &Hparams, l: usize) -> Result<LayerDump, GateError> {
        let (n, ff) = (hp.n_embd as u64, hp.experts.ff as u64);
        let (ne, nu) = (N_EXPERT as u64, N_USED as u64);
        let post_a = format!("hc_attn_post-{l}");
        let fold_name = format!("hc_ffn_pre-{l}");
        let norm = format!("ffn_norm-{l}");
        let streams = values(
            set,
            row(set, &post_a, [n, 4, 1, 1], "HC_POST", (None, None))?,
        )?;
        let fold_in = values(
            set,
            row(
                set,
                &fold_name,
                [n, 1, 1, 1],
                "MUL_MULTI_ADD",
                (Some(post_a.as_str()), None),
            )?,
        )?;
        let x = values(
            set,
            row(
                set,
                &norm,
                [n, 1, 1, 1],
                "FUSED_RMS_NORM",
                (Some(fold_name.as_str()), Some(names::ffn_norm(l).as_str())),
            )?,
        )?;
        let logits_name = format!("ffn_moe_logits-{l}");
        let logits = values(
            set,
            row(
                set,
                &logits_name,
                [ne, 1, 1, 1],
                "MUL_MAT",
                (Some(names::ffn_gate_inp(l).as_str()), Some(norm.as_str())),
            )?,
        )?;
        let probs_name = format!("ffn_moe_probs-{l}");
        row(
            set,
            &probs_name,
            [ne, 1, 1, 1],
            "SQRT_SOFTPLUS",
            (Some(logits_name.as_str()), None),
        )?;
        let probs_biased = values(
            set,
            row(
                set,
                &format!("ffn_moe_probs_biased-{l}"),
                [ne, 1, 1, 1],
                "ADD",
                (
                    Some(probs_name.as_str()),
                    Some(names::exp_probs_b(l).as_str()),
                ),
            )?,
        )?;
        let topk = set.man.tensor(&format!("ffn_moe_topk-{l}"), 0)?;
        let ids_v = topk_ids_logical_within(&set.man, topk, u32::try_from(N_EXPERT)?)?;
        let ids: [u32; N_USED] = ids_v
            .iter()
            .map(|&i| i.cast_unsigned())
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|v: Vec<u32>| format!("{}: {} ids at layer {l}", set.name, v.len()))?;
        let ws_name = format!("ffn_moe_weights_scaled-{l}");
        let w_v = values(
            set,
            row(set, &ws_name, [1, nu, 1, 1], "SCALE", (None, None))?,
        )?;
        let w: [f32; N_USED] = w_v
            .try_into()
            .map_err(|_| format!("{}: {ws_name} is not {N_USED} weights", set.name))?;
        let par = format!("ffn_moe_gate_par-{l}");
        let h = values(
            set,
            row(
                set,
                &par,
                [ff, nu, 1, 1],
                "MOE_FUSED_UP_GATE",
                (
                    Some(names::ffn_up_exps(l).as_str()),
                    Some(names::ffn_gate_exps(l).as_str()),
                ),
            )?,
        )?;
        let down_name = format!("ffn_moe_down-{l}");
        let down = values(
            set,
            row(
                set,
                &down_name,
                [n, nu, 1, 1],
                "MUL_MAT_ID",
                (Some(names::ffn_down_exps(l).as_str()), Some(par.as_str())),
            )?,
        )?;
        let moe_name = format!("ffn_moe_out-{l}");
        row(
            set,
            &moe_name,
            [n, 1, 1, 1],
            "MUL_MULTI_ADD",
            (Some(down_name.as_str()), Some(ws_name.as_str())),
        )?;
        let upg = format!("ffn_up_gate-{l}");
        let sh_h = values(
            set,
            row(
                set,
                &upg,
                [ff, 1, 1, 1],
                "FUSED_UP_GATE",
                (
                    Some(names::ffn_up_shexp(l).as_str()),
                    Some(names::ffn_gate_shexp(l).as_str()),
                ),
            )?,
        )?;
        let sh_name = format!("ffn_shexp-{l}");
        let sh = values(
            set,
            row(
                set,
                &sh_name,
                [n, 1, 1, 1],
                "MUL_MAT",
                (Some(names::ffn_down_shexp(l).as_str()), Some(upg.as_str())),
            )?,
        )?;
        let out_name = format!("ffn_out-{l}");
        let y = values(
            set,
            row(
                set,
                &out_name,
                [n, 1, 1, 1],
                "ADD",
                (Some(moe_name.as_str()), Some(sh_name.as_str())),
            )?,
        )?;
        // The ffn HC_PRE: the node reading this layer's mixes and the ffn
        // scale; its mixes read hc_ffn_fn over the streams above.
        let mixes_name = format!("hc_pre_mixes-{l}");
        let scale_name = names::hc_ffn_scale(l);
        let node = set
            .man
            .tensors
            .iter()
            .find(|r| {
                r.op == "HC_PRE"
                    && src0(r) == mixes_name
                    && r.src1.as_deref() == Some(scale_name.as_str())
            })
            .ok_or_else(|| {
                format!(
                    "{}: no HC_PRE reads {mixes_name} with {scale_name}",
                    set.name
                )
            })?;
        node.expect(&node.name, "f32", [HC_MIX as u64, 1, 1, 1], "HC_PRE")?;
        let mixes = set.man.tensor(&mixes_name, 1)?;
        let rms = set.man.tensor(&format!("hc_pre-{l}"), 1)?;
        if src0(mixes) != names::hc_ffn_fn(l) || src0(rms) != format!("{post_a} (reshaped)") {
            return Err(format!(
                "{}: the ffn HC_PRE at layer {l} reads {} over {}, not hc_ffn_fn over {post_a}",
                set.name,
                src0(mixes),
                src0(rms)
            )
            .into());
        }
        let hc: [f32; HC_MIX] = values(set, node)?
            .try_into()
            .map_err(|_| format!("{}: {} is not {HC_MIX} values", set.name, node.name))?;
        let l_name = format!("l_out-{l}");
        let l_out = values(
            set,
            row(
                set,
                &l_name,
                [n, 4, 1, 1],
                "HC_POST",
                (Some(out_name.as_str()), None),
            )?,
        )?;
        let fold_out = if l + 1 < hp.n_layer {
            let f = format!("hc_attn_pre-{}", l + 1);
            let r = row(set, &f, [n, 1, 1, 1], "MUL_MULTI_ADD", (None, None))?;
            let engram = format!("engram_out-{}", l + 1);
            if src0(r) == l_name {
                Some(values(set, r)?)
            } else if src0(r) == engram {
                None
            } else {
                return Err(format!(
                    "{}: {f} reads {}, neither {l_name} nor {engram}",
                    set.name,
                    src0(r)
                )
                .into());
            }
        } else {
            None
        };
        Ok(LayerDump {
            streams,
            fold_in,
            x,
            logits,
            probs_biased,
            ids,
            w,
            h,
            down,
            sh_h,
            sh,
            y,
            hc,
            l_out,
            fold_out,
        })
    }

    // ---------------------------------------------------------- the file

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

    fn tensor_type(split: &Split, name: &str) -> Result<GgmlType, GateError> {
        Ok(split
            .find(name)
            .map(|(_, t)| t.ty)
            .ok_or_else(|| format!("tensor {name} is not in the model"))?)
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

    /// Rows `r0 .. r0 + n` of a tensor of `k`-value rows of type `ty`,
    /// decoded to f32 by `dequant_row`.
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

    /// One output row of a dot in f64 against two activations (the MoE
    /// gate's): ours, ik's, and each one's sum of absolute terms.
    #[derive(Clone, Copy, Default)]
    struct RowStat {
        ours: f64,
        ik: f64,
        abs_ours: f64,
        abs_ik: f64,
    }

    /// Every row of `w` (rows of `xo.len()` values) against both activations.
    /// The products of two f32 are exact in f64.
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

    // ------------------------------------------------ the routing's band

    /// One f32 ulp at `v`.
    fn ulp32(v: f64) -> f64 {
        let a = v.abs() as f32;
        if a < f32::MIN_POSITIVE {
            return f64::from(f32::from_bits(1));
        }
        f64::from(f32::from_bits(a.to_bits() & 0x7f80_0000)) * f64::from(f32::EPSILON)
    }

    /// The exact `sqrt(ln(1 + e^x))`.
    fn softplus64(x: f32) -> f64 {
        let x = f64::from(x);
        if x > 20.0 {
            x.sqrt()
        } else {
            (1.0 + x.exp()).ln().sqrt()
        }
    }

    /// How far one side's f32 score can sit from the exact score at `x`
    /// (the MoE gate's derivation).
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

    /// Whether the dump's ordered top six could come out otherwise from our
    /// logits: some expert among the first six of ik's order and one after it
    /// in that order sit within the sum of their selection bands.
    /// Returns the smallest gap over band sum it saw (above 1: decided).
    fn tie_margin(dump: &LayerDump, logits: &[f32], bias: &[f32], probs: &[f32]) -> f64 {
        let band: Vec<f64> = (0..N_EXPERT)
            .map(|e| {
                let (lo, lk) = (logits[e], dump.logits[e]);
                let sel_o = f64::from(probs[e] + bias[e]);
                (softplus64(lo) - softplus64(lk)).abs()
                    + softplus_err(lo, DEVICE_EXP_ULPS, DEVICE_LOG_ULPS)
                    + softplus_err(lk, HOST_EXP_ULPS, HOST_LOG_ULPS)
                    + 0.5 * ulp32(sel_o)
                    + 0.5 * ulp32(f64::from(dump.probs_biased[e]))
            })
            .collect();
        let v = &dump.probs_biased;
        let mut order: Vec<usize> = (0..N_EXPERT).collect();
        order.sort_by(|&a, &b| (v[b] + 0.0).total_cmp(&(v[a] + 0.0)).then(b.cmp(&a)));
        let mut worst = f64::INFINITY;
        for p in 0..N_USED {
            let a = order[p];
            for &b in &order[p + 1..] {
                let gap = f64::from(v[a]) - f64::from(v[b]);
                worst = worst.min(gap / (band[a] + band[b]));
            }
        }
        worst
    }

    /// How far the two engines' norms of the same folded input can sit
    /// apart, relative to the value: each side's scale rounds at most
    /// `γ(k + 4)` through its sum of `k` squares, the mean, the `+ eps`, the
    /// root and the reciprocal (half of it through the root), and each value
    /// takes two products. Summed over both sides.
    fn norm_band(k: usize) -> f64 {
        gamma(k + 4) + 2.0 * gamma(4)
    }

    /// Whether every value of `a` is within `delta` of `b`'s, relative to the
    /// larger magnitude: the router read the dump's input up to the norms'
    /// rounding, so a selection that differs can be a near tie and not a
    /// wrong input.
    fn x_within(a: &[f32], b: &[f32], delta: f64) -> bool {
        a.iter().zip(b).all(|(&x, &y)| {
            let (x, y) = (f64::from(x), f64::from(y));
            (x - y).abs() <= delta * x.abs().max(y.abs())
        })
    }

    // ------------------------------------------------------ the slot map

    /// The gate's slot map and, per layer, the experts its card stacks hold in
    /// slot order (module doc).
    fn synthetic_map(
        split: &Split,
        hp: &Hparams,
        ids: &[Vec<[u32; N_USED]>],
    ) -> Result<(SlotMap, Vec<Vec<u32>>), GateError> {
        let n_expert = hp.experts.n_expert;
        let mut rows = vec![HOST; hp.n_layer * n_expert];
        let mut cards = Vec::with_capacity(hp.n_layer);
        for (l, per_set) in ids.iter().enumerate() {
            if tensor_type(split, &names::ffn_down_exps(l))? != GgmlType::Q4_K {
                cards.push(Vec::new());
                continue;
            }
            // Per expert: Some(true) card, Some(false) host.
            let mut side: Vec<Option<bool>> = vec![None; n_expert];
            for tok in per_set {
                for (rank, &e) in tok.iter().enumerate() {
                    let s = &mut side[e as usize];
                    if s.is_none() {
                        *s = Some(rank % 2 == 0);
                    }
                }
            }
            for tok in per_set {
                if !has_side(&side, tok, false) {
                    side[tok[N_USED - 1] as usize] = Some(false);
                }
                if !has_side(&side, tok, true) {
                    side[tok[0] as usize] = Some(true);
                }
            }
            for tok in per_set {
                if !(has_side(&side, tok, true) && has_side(&side, tok, false)) {
                    return Err(format!(
                        "layer {l}: no split of the sets' ids {per_set:?} puts ids of every token \
                         on both sides"
                    )
                    .into());
                }
            }
            // Slots against the id order: the largest id first.
            let mut card: Vec<u32> = (0..n_expert as u32)
                .filter(|&e| side[e as usize] == Some(true))
                .collect();
            card.reverse();
            for (slot, &e) in card.iter().enumerate() {
                rows[l * n_expert + e as usize] = u32::try_from(slot)?;
            }
            cards.push(card);
        }
        Ok((SlotMap::from_rows(0..hp.n_layer, n_expert, rows)?, cards))
    }

    /// Whether one of `tok`'s ids is on the side `card` names.
    fn has_side(side: &[Option<bool>], tok: &[u32; N_USED], card: bool) -> bool {
        tok.iter().any(|&e| side[e as usize] == Some(card))
    }

    /// The routed stacks of `experts` (slot order), compacted and uploaded.
    struct Stacks {
        gate: DeviceTensor<u32>,
        up: DeviceTensor<u32>,
        down: DeviceTensor<u32>,
    }

    fn upload_stacks(
        split: &Split,
        stream: &CudaStream,
        hp: &Hparams,
        l: usize,
        experts: &[u32],
    ) -> Result<Stacks, GateError> {
        let (n, ff, ne) = (hp.n_embd, hp.experts.ff, hp.experts.n_expert as u64);
        let gu = |name: String| tensor(split, &name, GgmlType::Q3_K, &[n as u64, ff as u64, ne]);
        let (g, u) = (gu(names::ffn_gate_exps(l))?, gu(names::ffn_up_exps(l))?);
        let d = tensor(
            split,
            &names::ffn_down_exps(l),
            GgmlType::Q4_K,
            &[ff as u64, n as u64, ne],
        )?;
        let (rb_gu, rb_d) = (
            row_bytes(GgmlType::Q3_K, n)?,
            row_bytes(GgmlType::Q4_K, ff)?,
        );
        let pack = |b: &[u8], rows: usize, rb: usize| -> Vec<u32> {
            let mut out = Vec::with_capacity(experts.len() * rows * rb);
            for &e in experts {
                let e = e as usize;
                out.extend_from_slice(&b[e * rows * rb..(e + 1) * rows * rb]);
            }
            bytes_to_words(&out)
        };
        let k = experts.len();
        Ok(Stacks {
            gate: DeviceTensor::upload(stream, &pack(g, ff, rb_gu), k * ff, rb_gu / 4)?,
            up: DeviceTensor::upload(stream, &pack(u, ff, rb_gu), k * ff, rb_gu / 4)?,
            down: DeviceTensor::upload(stream, &pack(d, n, rb_d), k * n, rb_d / 4)?,
        })
    }

    /// Layer `l`'s dense tensors the piece reads, made resident from the file.
    fn layer_weights(stream: &CudaStream, split: &Split, l: usize) -> Result<Weights, GateError> {
        let keep = [
            names::ffn_norm(l),
            names::ffn_gate_inp(l),
            names::exp_probs_b(l),
            names::ffn_gate_shexp(l),
            names::ffn_up_shexp(l),
            names::ffn_down_shexp(l),
            names::hc_ffn_fn(l),
            names::hc_ffn_scale(l),
            names::hc_ffn_base(l),
        ];
        let w = Weights::load_where(stream, split, |name| keep.iter().any(|k| k == name))?;
        let missing: Vec<&String> = keep.iter().filter(|k| w.get(k).is_none()).collect();
        if !missing.is_empty() {
            return Err(format!("layer {l}: not in the file: {missing:?}").into());
        }
        Ok(w)
    }

    fn f32_buf<'w>(w: &'w Weights, name: &str) -> Result<&'w DeviceBuffer<f32>, GateError> {
        match w.get(name) {
            Some(DevWeight::F32 { w, .. }) => Ok(w.buf()),
            _ => Err(format!("{name} is not a resident f32 tensor").into()),
        }
    }

    fn f32_ten<'w>(w: &'w Weights, name: &str) -> Result<&'w DeviceTensor<f32>, GateError> {
        match w.get(name) {
            Some(DevWeight::F32 { w, .. }) => Ok(w),
            _ => Err(format!("{name} is not a resident f32 tensor").into()),
        }
    }

    // ---------------------------------------------------- the op path

    /// The gate's own kernels and buffers: the op path the piece must equal.
    struct Op {
        fused: FusedKernels,
        router: RouterKernels,
        experts: ExpertKernels,
        hc: HcKernels,
        x: DeviceBuffer<f32>,
        act_x: Q8Act,
        rout: RouterOut,
        sel: DeviceBuffer<u32>,
        h: DeviceBuffer<f32>,
        act_h: Q8Act,
        down: DeviceBuffer<f32>,
        sh_h: DeviceBuffer<f32>,
        sh_y: DeviceBuffer<f32>,
        mixes: DeviceBuffer<f32>,
        hc_out: DeviceBuffer<f32>,
        hc_scratch: HcPreScratch,
        host: HostScratch,
    }

    impl Op {
        fn new(gpu: &Gpu, hp: &Hparams) -> Result<Op, GateError> {
            let (ctx, stream) = (gpu.context(), gpu.stream());
            let (n, ff) = (hp.n_embd, hp.experts.ff);
            let z = |k: usize| DeviceBuffer::<f32>::zeroed(stream, k);
            Ok(Op {
                fused: FusedKernels::load(ctx)?,
                router: RouterKernels::load(ctx)?,
                experts: ExpertKernels::load(ctx)?,
                hc: HcKernels::load(ctx)?,
                x: z(n)?,
                act_x: Q8Act::with_k(stream, 1, n)?,
                rout: RouterOut::new(stream)?,
                sel: DeviceBuffer::zeroed(stream, N_USED)?,
                h: z(N_USED * ff)?,
                act_h: Q8Act::with_k(stream, N_USED, ff)?,
                down: z(N_USED * n)?,
                sh_h: z(ff)?,
                sh_y: z(n)?,
                mixes: z(HC_MIX)?,
                hc_out: z(HC_MIX)?,
                hc_scratch: HcPreScratch::new(stream, HC_STREAMS * n)?,
                host: HostScratch::new(n, ff),
            })
        }
    }

    /// What one run leaves: every buffer the pins read, on the host.
    #[derive(Clone)]
    struct Out {
        x: Vec<f32>,
        logits: Vec<f32>,
        probs: Vec<f32>,
        ids: Vec<u32>,
        w: Vec<f32>,
        sel: Vec<u32>,
        /// The card's slots of h and down, in slot order (others zero).
        h: Vec<f32>,
        down: Vec<f32>,
        sh_h: Vec<f32>,
        sh: Vec<f32>,
        hsum: Vec<f32>,
        y: Vec<f32>,
        hc: Vec<f32>,
        streams: Vec<f32>,
        fold: Option<Vec<f32>>,
    }

    /// Each field of `a` and `b`, in order, and whether its bits agree.
    fn same(a: &Out, b: &Out) -> Vec<(&'static str, bool)> {
        let u = |x: &[u32], y: &[u32]| x == y;
        vec![
            ("x", bits_equal(&a.x, &b.x)),
            ("logits", bits_equal(&a.logits, &b.logits)),
            ("probs", bits_equal(&a.probs, &b.probs)),
            ("ids", u(&a.ids, &b.ids)),
            ("weights", bits_equal(&a.w, &b.w)),
            ("sel", u(&a.sel, &b.sel)),
            ("h", bits_equal(&a.h, &b.h)),
            ("down", bits_equal(&a.down, &b.down)),
            ("shexp_h", bits_equal(&a.sh_h, &b.sh_h)),
            ("shexp", bits_equal(&a.sh, &b.sh)),
            ("hsum", bits_equal(&a.hsum, &b.hsum)),
            ("y", bits_equal(&a.y, &b.y)),
            ("hc", bits_equal(&a.hc, &b.hc)),
            ("streams", bits_equal(&a.streams, &b.streams)),
            (
                "fold",
                match (&a.fold, &b.fold) {
                    (Some(x), Some(y)) => bits_equal(x, y),
                    (None, None) => true,
                    _ => false,
                },
            ),
        ]
    }

    /// The first field that differs, `-` when none does.
    fn first_diff(a: &Out, b: &Out) -> &'static str {
        same(a, b)
            .into_iter()
            .find(|(_, ok)| !ok)
            .map_or("-", |(name, _)| name)
    }

    /// Keep the card's slots of a slot-major buffer (`k` per slot), zero the
    /// rest: a host slot's rows are whatever an earlier launch left.
    fn card_slots(v: &[f32], sel: &[u32], n_card: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; v.len()];
        for (j, &p) in sel.iter().enumerate() {
            if (p as usize) < n_card {
                out[j * k..(j + 1) * k].copy_from_slice(&v[j * k..(j + 1) * k]);
            }
        }
        out
    }

    /// The HC_POST of one value (the HC gate's transcription): `hc` in the
    /// kernel layout.
    fn post_elem(x: f32, r: [f32; 4], hc: &[f32]) -> [f32; 4] {
        let mut o = [0.0f32; 4];
        for (i, oi) in o.iter_mut().enumerate() {
            let mut s = x.mul_add(hc[4 + i], hc[8 + i] * r[0]);
            for (j, &rj) in r.iter().enumerate().skip(1) {
                s = hc[8 + 4 * j + i].mul_add(rj, s);
            }
            *oi = s;
        }
        o
    }

    /// The fold of four stream values by `pre`.
    fn fold_elem(o: [f32; 4], pre: &[f32]) -> f32 {
        let mut y = o[0] * pre[0];
        for (&oj, &pj) in o.iter().zip(pre).skip(1) {
            y = oj.mul_add(pj, y);
        }
        y
    }

    fn streams_at(s: &[f32], n: usize, d: usize) -> [f32; 4] {
        [s[d], s[d + n], s[d + 2 * n], s[d + 3 * n]]
    }

    /// Everything the op path needs for one layer.
    struct LayerCx<'a> {
        l: usize,
        w: &'a Weights,
        stacks: Option<&'a Stacks>,
        row: &'a [u32],
        n_card: usize,
        host: Option<&'a HostLayer>,
        fold: bool,
    }

    /// Per host slot of the op path's service: its list index, id and
    /// weight, and the host's h and down for it.
    struct HostSlot {
        slot: usize,
        h: Vec<f32>,
        down: Vec<f32>,
    }

    /// The op path for the injected inputs already in `streams`/`fold_in`.
    fn op_path(
        gpu: &Gpu,
        op: &mut Op,
        split: &Split,
        hp: &Hparams,
        lc: &LayerCx<'_>,
        streams: &DeviceBuffer<f32>,
        fold_in: &DeviceBuffer<f32>,
    ) -> Result<(Out, Vec<HostSlot>), GateError> {
        let stream = gpu.stream();
        let (n, ff, l, w) = (hp.n_embd, hp.experts.ff, lc.l, lc.w);
        let kind = &hp.layers[l];
        op.fused.enqueue_norm_quant(
            stream,
            fold_in,
            f32_buf(w, &names::ffn_norm(l))?,
            hp.rms_eps,
            &mut op.act_x,
            &mut op.x,
        )?;
        op.router.enqueue_router(
            stream,
            f32_ten(w, &names::ffn_gate_inp(l))?,
            &op.x,
            f32_buf(w, &names::exp_probs_b(l))?,
            hp.experts.routed_scale,
            &mut op.rout,
        )?;
        let ids = op.rout.ids.to_host_vec(stream)?;
        let weights = op.rout.weights.to_host_vec(stream)?;
        let sel: Vec<u32> = ids
            .iter()
            .map(|&id| lc.row.get(id as usize).copied().unwrap_or(HOST))
            .collect();
        op.sel.copy_from_host(stream, &sel)?;
        if let Some(s) = lc.stacks {
            let args = ExpertGateUp {
                wg: &s.gate,
                wu: &s.up,
                act: &op.act_x,
                sel: &op.sel,
                n_slots: N_USED,
                rows_per_expert: ff,
                limit: kind.swiglu_limit,
            };
            op.experts
                .enqueue_expert_gate_up(stream, &args, &mut op.h)?;
            gpu.enqueue_quantize_q8_1(&op.h, &mut op.act_h)?;
            gpu.q4k_sel().enqueue_gemv_q4k_sel(
                stream,
                &s.down,
                &op.act_h,
                &op.sel,
                N_USED,
                n,
                &mut op.down,
            )?;
        }
        let (Some(sg), Some(su), Some(DevWeight::Q8_0 { qs, d, .. })) = (
            w.get(&names::ffn_gate_shexp(l)),
            w.get(&names::ffn_up_shexp(l)),
            w.get(&names::ffn_down_shexp(l)),
        ) else {
            return Err(format!("layer {l}: the shared expert is not resident as q8_0").into());
        };
        op.experts.enqueue_shexp_gate_up(
            stream,
            sg,
            su,
            &op.x,
            kind.swiglu_limit_shared,
            &mut op.sh_h,
        )?;
        gpu.q8f32()
            .enqueue_q8_0_gemv(stream, qs, d, &op.sh_h, 1, &mut op.sh_y)?;
        let Some(DevWeight::KQuant { w: hc_w, .. }) = w.get(&names::hc_ffn_fn(l)) else {
            return Err(format!("layer {l}: hc_ffn_fn is not resident as a K-quant").into());
        };
        let params = HcParams {
            w: hc_w,
            scale: f32_buf(w, &names::hc_ffn_scale(l))?,
            base: f32_buf(w, &names::hc_ffn_base(l))?,
            eps: hp.hc.eps,
            iters: u32::try_from(hp.hc.sinkhorn_iters)?,
        };
        let pre = HcPreArgs {
            params: &params,
            x: streams,
            tokens: 1,
            rms_eps: hp.rms_eps,
        };
        op.hc.enqueue_pre(
            stream,
            &pre,
            &mut op.hc_scratch,
            &mut op.mixes,
            &mut op.hc_out,
        )?;
        stream.synchronize()?;

        let x = op.x.to_host_vec(stream)?;
        let h = card_slots(&op.h.to_host_vec(stream)?, &sel, lc.n_card, ff);
        let down = card_slots(&op.down.to_host_vec(stream)?, &sel, lc.n_card, n);
        let sh_h = op.sh_h.to_host_vec(stream)?;
        let sh = op.sh_y.to_host_vec(stream)?;
        let hc = op.hc_out.to_host_vec(stream)?;
        let s_in = streams.to_host_vec(stream)?;

        // The host's share: the slots the map sends to the host, in slot
        // order, with the op path's own activation.
        let list: Vec<(u32, f32)> = (0..N_USED)
            .filter(|&j| sel[j] as usize >= lc.n_card)
            .map(|j| (ids[j], weights[j]))
            .collect();
        let slots: Vec<usize> = (0..N_USED)
            .filter(|&j| sel[j] as usize >= lc.n_card)
            .collect();
        let mut hsum = vec![0.0f32; n];
        let mut host_slots = Vec::with_capacity(list.len());
        let hl = lc
            .host
            .ok_or_else(|| format!("layer {l}: no host view of a routed layer"))?;
        let xt = Tensor2::from_vec(n, 1, x.clone());
        hl.experts_into(split, &xt, &list, &mut hsum, &mut op.host)?;
        for (i, &slot) in slots.iter().enumerate() {
            host_slots.push(HostSlot {
                slot,
                h: op.host.par(i).data.clone(),
                down: op.host.down(i).data.clone(),
            });
        }

        let card: [bool; N_USED] = std::array::from_fn(|j| (sel[j] as usize) < lc.n_card);
        let wv: [f32; N_USED] = std::array::from_fn(|j| if card[j] { weights[j] } else { 0.0 });
        let y: Vec<f32> = (0..n)
            .map(|dd| {
                let dv: [f32; N_USED] =
                    std::array::from_fn(|j| if card[j] { down[j * n + dd] } else { 0.0 });
                combine_elem(dv, wv, card, hsum[dd], sh[dd])
            })
            .collect();
        let mut st = vec![0.0f32; HC_STREAMS * n];
        let mut fold = lc.fold.then(|| vec![0.0f32; n]);
        for dd in 0..n {
            let o = post_elem(y[dd], streams_at(&s_in, n, dd), &hc);
            for (i, &oi) in o.iter().enumerate() {
                st[i * n + dd] = oi;
            }
            if let Some(f) = fold.as_mut() {
                f[dd] = fold_elem(o, &hc[..4]);
            }
        }
        let out = Out {
            x,
            logits: op.rout.logits.to_host_vec(stream)?,
            probs: op.rout.probs.to_host_vec(stream)?,
            ids,
            w: weights,
            sel,
            h,
            down,
            sh_h,
            sh,
            hsum,
            y,
            hc,
            streams: st,
            fold,
        };
        Ok((out, host_slots))
    }

    // ------------------------------------------------------ the piece

    /// The device buffers the piece shares with the step, the gate's own.
    struct Io {
        streams: DeviceBuffer<f32>,
        fold_in: DeviceBuffer<f32>,
        streams_out: DeviceBuffer<f32>,
        fold_out: DeviceBuffer<f32>,
        slots: DeviceTensor<u32>,
    }

    /// One eager run of the piece at layer `lc.l`, its buffers read back.
    fn run_piece(
        gpu: &Gpu,
        piece: &mut FfnPiece,
        hybrid: &mut Hybrid<Ds41Host>,
        io: &mut Io,
        lc: &LayerCx<'_>,
    ) -> Result<Out, GateError> {
        let stream = gpu.stream();
        hybrid.begin_chain(stream)?;
        enqueue(gpu, piece, hybrid, io, lc)?;
        stream.synchronize()?;
        read_piece(gpu, piece, hybrid, io, lc)
    }

    fn enqueue(
        gpu: &Gpu,
        piece: &mut FfnPiece,
        hybrid: &mut Hybrid<Ds41Host>,
        io: &mut Io,
        lc: &LayerCx<'_>,
    ) -> Result<(), bloomery_gpu::GpuError> {
        let card = lc.stacks.map(|s| CardStacks {
            gate: &s.gate,
            up: &s.up,
            down: &s.down,
        });
        let fio = FfnIo {
            streams: &io.streams,
            fold_in: &io.fold_in,
            streams_out: &mut io.streams_out,
            fold_out: lc.fold.then_some(&mut io.fold_out),
            slots: &io.slots,
        };
        piece.enqueue(gpu, lc.w, card, fio, hybrid, lc.l)
    }

    fn read_piece(
        gpu: &Gpu,
        piece: &FfnPiece,
        hybrid: &Hybrid<Ds41Host>,
        io: &Io,
        lc: &LayerCx<'_>,
    ) -> Result<Out, GateError> {
        let stream = gpu.stream();
        let t = piece.taps();
        let sel = t.sel.to_host_vec(stream)?;
        let (n, ff) = (io.fold_in.len(), t.shexp_h.len());
        Ok(Out {
            x: hybrid.boundary().normed().to_host_vec(stream)?,
            logits: t.router.logits.to_host_vec(stream)?,
            probs: t.router.probs.to_host_vec(stream)?,
            ids: t.router.ids.to_host_vec(stream)?,
            w: t.router.weights.to_host_vec(stream)?,
            h: card_slots(&t.h.to_host_vec(stream)?, &sel, lc.n_card, ff),
            down: card_slots(&t.down.to_host_vec(stream)?, &sel, lc.n_card, n),
            sel,
            sh_h: t.shexp_h.to_host_vec(stream)?,
            sh: t.shexp.to_host_vec(stream)?,
            hsum: hybrid.hsum_copy()?,
            y: t.y.to_host_vec(stream)?,
            hc: t.hc.to_host_vec(stream)?,
            streams: io.streams_out.to_host_vec(stream)?,
            fold: if lc.fold {
                Some(io.fold_out.to_host_vec(stream)?)
            } else {
                None
            },
        })
    }

    // ------------------------------------------------------- the bands

    /// A pin per value (module doc): `pred`, our output minus ik's in exact
    /// arithmetic from each side's own inputs, and `bound`, both sides' float
    /// rounding around it.
    struct Pred {
        pred: Vec<f64>,
        bound: Vec<f64>,
    }

    /// The largest `|(a - b) - pred| / bound` (0 where the gap is 0,
    /// infinite where only the bound is).
    fn gap_ratio(a: &[f32], b: &[f32], p: &Pred) -> f64 {
        a.iter()
            .zip(b)
            .zip(p.pred.iter().zip(&p.bound))
            .map(|((&x, &y), (&pr, &bd))| {
                let gap = ((f64::from(x) - f64::from(y)) - pr).abs();
                if gap == 0.0 {
                    0.0
                } else if bd > 0.0 {
                    gap / bd
                } else {
                    f64::INFINITY
                }
            })
            .fold(
                0.0f64,
                |m, r| if r.is_nan() { f64::INFINITY } else { m.max(r) },
            )
    }

    /// `max|pred| / max|of|`: how large a difference the prediction carries.
    fn pred_rel(p: &Pred, of: &[f32]) -> f64 {
        let num = p.pred.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        let den = of.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
        if den > 0.0 { num / den } else { num }
    }

    /// `max|a - b| / max|b|`.
    fn rel(a: &[f32], b: &[f32]) -> f64 {
        let den = b.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
        let num = a.iter().zip(b).fold(0.0f64, |m, (&x, &y)| {
            m.max((f64::from(x) - f64::from(y)).abs())
        });
        if den > 0.0 { num / den } else { num }
    }

    /// The host-side data the bands read at one layer: each routed expert's
    /// down rows the sets use, and the shared down.
    struct BandWeights {
        down_ty: GgmlType,
        /// Expert id to its down rows (`n_embd` rows of `ff`).
        downs: Vec<(u32, Vec<f32>)>,
        sh_down: Vec<f32>,
    }

    fn band_weights(
        split: &Split,
        hp: &Hparams,
        l: usize,
        experts: &[u32],
    ) -> Result<BandWeights, GateError> {
        let (n, ff, ne) = (hp.n_embd, hp.experts.ff, hp.experts.n_expert as u64);
        let down_name = names::ffn_down_exps(l);
        let down_ty = tensor_type(split, &down_name)?;
        let bytes = tensor(split, &down_name, down_ty, &[ff as u64, n as u64, ne])?;
        let mut downs = Vec::with_capacity(experts.len());
        for &e in experts {
            downs.push((e, rows_f32(down_ty, bytes, e as usize * n, n, ff)?));
        }
        let sb = tensor(
            split,
            &names::ffn_down_shexp(l),
            GgmlType::Q8_0,
            &[ff as u64, n as u64],
        )?;
        Ok(BandWeights {
            down_ty,
            downs,
            sh_down: rows_f32(GgmlType::Q8_0, sb, 0, n, ff)?,
        })
    }

    /// The combine's pin per value (module doc), from our run `o` (its host
    /// slots' rows in `hs`), ik's dump `d` and the down weights.
    fn pred_y(
        o: &Out,
        hs: &[HostSlot],
        d: &LayerDump,
        bw: &BandWeights,
        n_card: usize,
    ) -> Result<Pred, GateError> {
        let n = d.y.len();
        let ff = d.sh_h.len();
        let decode = if bw.down_ty == GgmlType::Q8_0 {
            0.0
        } else {
            KQ_DECODE
        };
        let ik_n = n_ik_quant(ff) + decode;
        let mut pred = vec![0.0f64; n];
        let mut bound = vec![0.0f64; n];
        // Per value: the magnitudes our card chain, the host's sum and ik's
        // sum add, and those the gate's own f64 sums add.
        let mut sum_o = vec![0.0f64; n];
        let mut sum_host = vec![0.0f64; n];
        let mut sum_k = vec![0.0f64; n];
        let mut mag64 = vec![0.0f64; n];
        let n_host = hs.len();
        for j in 0..N_USED {
            let e = o.ids[j];
            let wd = bw
                .downs
                .iter()
                .find(|(id, _)| *id == e)
                .map(|(_, rows)| rows)
                .ok_or_else(|| format!("expert {e}'s down rows were not read"))?;
            let hk = &d.h[j * ff..(j + 1) * ff];
            let mut xi = vec![0.0f32; ff];
            quantize_activations(bw.down_ty, hk, &mut xi)?;
            let (wo, wk) = (f64::from(o.w[j]), f64::from(d.w[j]));
            let dk = &d.down[j * n..(j + 1) * n];
            let card = (o.sel[j] as usize) < n_card;
            let (xo, own_n, dv): (Vec<f32>, f64, Vec<f32>) = if card {
                (
                    q8_1_dequant(&o.h[j * ff..(j + 1) * ff], ff, 1),
                    n_ours(ff) + decode + Q8_IMAGE,
                    o.down[j * n..(j + 1) * n].to_vec(),
                )
            } else {
                let s = hs
                    .iter()
                    .find(|s| s.slot == j)
                    .ok_or_else(|| format!("slot {j} is neither the card's nor the host's"))?;
                let mut xh = vec![0.0f32; ff];
                quantize_activations(bw.down_ty, &s.h, &mut xh)?;
                (xh, n_ik_quant(ff) + decode, s.down.clone())
            };
            let st = row_stats(wd, &xo, &xi)?;
            for (dd, s) in st.iter().enumerate() {
                pred[dd] += wo * s.ours - wk * s.ik;
                bound[dd] += wo.abs() * own_n * U * s.abs_ours + wk.abs() * ik_n * U * s.abs_ik;
                mag64[dd] += wo.abs() * s.abs_ours + wk.abs() * s.abs_ik;
                let term = (wo * f64::from(dv[dd])).abs();
                if card {
                    sum_o[dd] += term;
                } else {
                    sum_host[dd] += term;
                }
                sum_k[dd] += (wk * f64::from(dk[dd])).abs();
            }
        }
        let mut xi = vec![0.0f32; ff];
        quantize_row_q8_2_x4_roundtrip(&d.sh_h, &mut xi);
        let st = row_stats(&bw.sh_down, &o.sh_h, &xi)?;
        let (g_o, g_h, g_k) = (
            gamma(n_card_slots(o, n_card) + 2),
            gamma(2 * n_host.max(1)),
            gamma(N_USED + 1),
        );
        let slack = (ff + N_USED + 2) as f64 * U64;
        for (dd, s) in st.iter().enumerate() {
            pred[dd] += s.ours - s.ik;
            bound[dd] += n_ours(ff) * U * s.abs_ours
                + n_ik_quant(ff) * U * s.abs_ik
                + g_o * (sum_o[dd] + f64::from(o.hsum[dd]).abs() + f64::from(o.sh[dd]).abs())
                + g_h * sum_host[dd]
                + g_k * (sum_k[dd] + f64::from(d.sh[dd]).abs())
                + slack * (mag64[dd] + s.abs_ours + s.abs_ik);
        }
        Ok(Pred { pred, bound })
    }

    fn n_card_slots(o: &Out, n_card: usize) -> usize {
        o.sel.iter().filter(|&&p| (p as usize) < n_card).count()
    }

    /// The streams' pin per (stream, value), `[4][n]`, carried from the
    /// combine's: HC_POST's exact difference at our combine as `yᵏ +
    /// pred_y` has it.
    fn pred_streams(o: &Out, d: &LayerDump, py: &Pred) -> Pred {
        let n = d.y.len();
        let (ho, hk) = (&o.hc, &d.hc);
        let mut pred = vec![0.0f64; HC_STREAMS * n];
        let mut bound = vec![0.0f64; HC_STREAMS * n];
        for dd in 0..n {
            let r = streams_at(&d.streams, n, dd);
            let (yo, yk) = (f64::from(o.y[dd]), f64::from(d.y[dd]));
            let y_pred = yk + py.pred[dd];
            for i in 0..HC_STREAMS {
                let (po, pk) = (f64::from(ho[4 + i]), f64::from(hk[4 + i]));
                let mut p = po * y_pred - pk * yk;
                let (mut mag_o, mut mag_k) = ((yo * po).abs(), (yk * pk).abs());
                for (j, &rj) in r.iter().enumerate() {
                    let (co, ck) = (f64::from(ho[8 + 4 * j + i]), f64::from(hk[8 + 4 * j + i]));
                    let rj = f64::from(rj);
                    p += (co - ck) * rj;
                    mag_o += (co * rj).abs();
                    mag_k += (ck * rj).abs();
                }
                pred[i * n + dd] = p;
                bound[i * n + dd] = po.abs() * py.bound[dd]
                    + gamma(5) * (mag_o + mag_k)
                    + 16.0 * U64 * (mag_o + mag_k + (po * y_pred).abs());
            }
        }
        Pred { pred, bound }
    }

    /// The fold's pin per value, carried from the streams'.
    fn pred_fold(o: &Out, d: &LayerDump, ps: &Pred) -> Pred {
        let n = d.y.len();
        let (ho, hk) = (&o.hc, &d.hc);
        let mut pred = vec![0.0f64; n];
        let mut bound = vec![0.0f64; n];
        for dd in 0..n {
            let (so, sk) = (streams_at(&o.streams, n, dd), streams_at(&d.l_out, n, dd));
            let (mut p, mut b) = (0.0f64, 0.0f64);
            let (mut mag_o, mut mag_k, mut mag64) = (0.0f64, 0.0f64, 0.0f64);
            for i in 0..HC_STREAMS {
                let (po, pk) = (f64::from(ho[i]), f64::from(hk[i]));
                let (ski, pi) = (f64::from(sk[i]), ps.pred[i * n + dd]);
                p += po * (ski + pi) - pk * ski;
                b += po.abs() * ps.bound[i * n + dd];
                mag_o += (f64::from(so[i]) * po).abs();
                mag_k += (ski * pk).abs();
                mag64 += (po * (ski + pi)).abs();
            }
            pred[dd] = p;
            bound[dd] = b + gamma(4) * (mag_o + mag_k) + 16.0 * U64 * (mag64 + mag_k);
        }
        Pred { pred, bound }
    }

    // -------------------------------------------------------- structure

    /// Node counts by kind: kernels, memcpy, batch memory operations, other.
    fn kinds(nodes: &[NodeInfo]) -> [usize; 4] {
        let mut c = [0usize; 4];
        for node in nodes {
            let i = match node.kind {
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL => 0,
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMCPY => 1,
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_BATCH_MEM_OP => 2,
                _ => 3,
            };
            c[i] += 1;
        }
        c
    }

    fn inject(stream: &CudaStream, io: &mut Io, d: &LayerDump) -> Result<(), GateError> {
        io.streams.copy_from_host(stream, &d.streams)?;
        io.fold_in.copy_from_host(stream, &d.fold_in)?;
        Ok(())
    }

    // ---------------------------------------------------------- driver

    #[derive(Default)]
    struct Tally {
        checks: u32,
        failed: u32,
        waived: u32,
    }

    impl Tally {
        fn add(&mut self, pass: bool) {
            self.checks += 1;
            if !pass {
                self.failed += 1;
            }
        }
    }

    pub fn run() -> Result<(), GateError> {
        let path = ref_model_path()?;
        let split = Split::open(&path)?;
        let hp = Hparams::read(&split)?;
        let gpu = Gpu::new()?;
        let stream = gpu.stream();
        let (n, ff) = (hp.n_embd, hp.experts.ff);
        println!(
            "gate_deepseek41_chain_ffn: device {} — layers {} n_embd {n} ff {ff} experts {} used {} \
             scale {} streams {}",
            gpu.device_name()?,
            hp.n_layer,
            hp.experts.n_expert,
            hp.experts.n_used,
            hp.experts.routed_scale,
            hp.hc.streams
        );
        let sets: Vec<SetData> = SETS
            .iter()
            .map(|&s| set_data(s))
            .collect::<Result<_, _>>()?;

        // Every layer's dump ids first: the map is built from both sets.
        let mut ids_per_layer: Vec<Vec<[u32; N_USED]>> = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let mut per = Vec::with_capacity(sets.len());
            for set in &sets {
                let topk = set.man.tensor(&format!("ffn_moe_topk-{l}"), 0)?;
                let v = topk_ids_logical_within(&set.man, topk, u32::try_from(N_EXPERT)?)?;
                let ids: [u32; N_USED] = v
                    .iter()
                    .map(|&i| i.cast_unsigned())
                    .collect::<Vec<_>>()
                    .try_into()
                    .map_err(|_| format!("{}: layer {l} ids", set.name))?;
                per.push(ids);
            }
            ids_per_layer.push(per);
        }
        let (map, cards) = synthetic_map(&split, &hp, &ids_per_layer)?;
        let host_layers: Vec<Option<HostLayer>> = (0..hp.n_layer)
            .map(|l| host::layer(&split, &hp, l))
            .collect::<Result<_, _>>()?;

        let shape = BoundaryShape {
            hidden: n,
            n_used: N_USED,
        };
        let boundary = Boundary::new(gpu.context(), stream, shape, map.clone(), true)?;
        let tier = Ds41Host::build(Split::open(&path)?, &hp, 0..hp.n_layer)?;
        let mut hybrid = Hybrid::new(boundary, tier, hp.n_layer)?;
        let off_boundary = Boundary::new(gpu.context(), stream, shape, map.clone(), false)?;
        let off_tier = Ds41Host::build(Split::open(&path)?, &hp, 0..hp.n_layer)?;
        let mut hybrid_off = Hybrid::new(off_boundary, off_tier, hp.n_layer)?;
        let mut piece = FfnPiece::new(&gpu, &hp, &map)?;
        let mut op = Op::new(&gpu, &hp)?;
        let mut io = Io {
            streams: DeviceBuffer::zeroed(stream, HC_STREAMS * n)?,
            fold_in: DeviceBuffer::zeroed(stream, n)?,
            streams_out: DeviceBuffer::zeroed(stream, HC_STREAMS * n)?,
            fold_out: DeviceBuffer::zeroed(stream, n)?,
            slots: DeviceTensor::upload(stream, map.as_slice(), hp.n_layer, hp.experts.n_expert)?,
        };
        println!(
            "piece: layers {:?} scratch_bytes={} map: {} layers with card experts, {} all-host",
            piece.layers(),
            piece.device_bytes(),
            cards.iter().filter(|c| !c.is_empty()).count(),
            cards.iter().filter(|c| c.is_empty()).count()
        );

        let mut tally = Tally::default();
        tally.add(no_local_depot(&[
            "ds41_ffn_handoff",
            "ds41_ffn_post",
            "ds41_ffn_post_streams",
        ])?);
        let mut overlap_done = false;
        for l in 0..hp.n_layer {
            let w = layer_weights(stream, &split, l)?;
            let stacks = if cards[l].is_empty() {
                None
            } else {
                Some(upload_stacks(&split, stream, &hp, l, &cards[l])?)
            };
            let dumps: Vec<LayerDump> = sets
                .iter()
                .map(|s| layer_dump(s, &hp, l))
                .collect::<Result<_, _>>()?;
            let fold = piece
                .folds(l)
                .ok_or_else(|| format!("layer {l} is outside the piece"))?;
            for (set, d) in sets.iter().zip(&dumps) {
                if d.fold_out.is_some() != fold {
                    return Err(format!(
                        "{}: layer {l}: the piece {} and the dump's next sub-layer {}",
                        set.name,
                        if fold {
                            "folds"
                        } else {
                            "ends with HC_POST alone"
                        },
                        if d.fold_out.is_some() {
                            "reads l_out"
                        } else {
                            "does not read l_out"
                        }
                    )
                    .into());
                }
            }
            let row = map
                .row(l)
                .ok_or_else(|| format!("layer {l} has no map row"))?;
            let lc = LayerCx {
                l,
                w: &w,
                stacks: stacks.as_ref(),
                row,
                n_card: map.on_card(l),
                host: host_layers[l].as_ref(),
                fold,
            };
            let mut used: Vec<u32> = dumps.iter().flat_map(|d| d.ids).collect();
            used.sort_unstable();
            used.dedup();
            let bw = band_weights(&split, &hp, l, &used)?;
            let bias = f32_buf(&w, &names::exp_probs_b(l))?.to_host_vec(stream)?;
            let mut eager = Vec::with_capacity(sets.len());
            for (set, d) in sets.iter().zip(&dumps) {
                inject(stream, &mut io, d)?;
                let got = run_piece(&gpu, &mut piece, &mut hybrid, &mut io, &lc)?;
                let (want, host_slots) =
                    op_path(&gpu, &mut op, &split, &hp, &lc, &io.streams, &io.fold_in)?;
                let diff = first_diff(&got, &want);
                let op_ok = diff == "-";

                // 3. Against the dump.
                let margin = tie_margin(d, &got.logits, &bias, &got.probs);
                let ids_eq = got.ids == d.ids;
                let x_rel = rel(&got.x, &d.x);
                let x_near = x_within(&got.x, &d.x, norm_band(n));
                let waived = !ids_eq && margin <= 1.0 && x_near;
                let kind = if lc.n_card > 0 {
                    format!("card={}", lc.n_card)
                } else {
                    "host-only".to_string()
                };
                let places: Vec<String> = got
                    .sel
                    .iter()
                    .map(|&p| {
                        if p == HOST {
                            "H".to_string()
                        } else {
                            p.to_string()
                        }
                    })
                    .collect();
                if !ids_eq {
                    // Other experts than the dump's: the pins below would
                    // compare different sums, so the line ends here.
                    if waived {
                        tally.waived += 1;
                    }
                    let pass = waived && op_ok;
                    println!(
                        "ffn set={} L={l} {kind} ids={:?} ik_ids={:?} tie_margin={margin:.3e} \
                         x_near={x_near} x_ik_rel={x_rel:.3e} op_path_bit={op_ok} first_diff={diff} \
                         — {} {}",
                        set.name,
                        got.ids,
                        d.ids,
                        if waived {
                            "near tie: ids and bands waived"
                        } else {
                            "the ids differ from the dump's"
                        },
                        verdict(pass)
                    );
                    tally.add(pass);
                    eager.push(got);
                    continue;
                }
                let py = pred_y(&got, &host_slots, d, &bw, lc.n_card)?;
                let ps = pred_streams(&got, d, &py);
                let y_ratio = gap_ratio(&got.y, &d.y, &py);
                let s_ratio = gap_ratio(&got.streams, &d.l_out, &ps);
                let (f_ratio, f_text) = match (&got.fold, &d.fold_out) {
                    (Some(f), Some(fk)) => {
                        let r = gap_ratio(f, fk, &pred_fold(&got, d, &ps));
                        (r, format!("{r:.3e}"))
                    }
                    _ => (0.0, "-".to_string()),
                };
                let pass = op_ok && ids_eq && y_ratio <= 1.0 && s_ratio <= 1.0 && f_ratio <= 1.0;
                println!(
                    "ffn set={} L={l} {kind} form={} places={} op_path_bit={op_ok} first_diff={diff} \
                     | ids_eq={ids_eq} tie_margin={margin:.3e} x_ik_rel={x_rel:.3e} | y_gap_ratio={y_ratio:.3e} \
                     y_pred_rel={:.3e} y_ik_rel={:.3e} streams_gap_ratio={s_ratio:.3e} \
                     streams_ik_rel={:.3e} fold_gap_ratio={f_text} {}",
                    set.name,
                    if fold { "post+fold" } else { "post" },
                    places.join(","),
                    pred_rel(&py, &d.y),
                    rel(&got.y, &d.y),
                    rel(&got.streams, &d.l_out),
                    verdict(pass)
                );
                tally.add(pass);
                eager.push(got);
            }

            // Structure: capture, replay per set with the host serving.
            let g = gpu.capture(|_| {
                hybrid.begin_chain(stream)?;
                enqueue(&gpu, &mut piece, &mut hybrid, &mut io, &lc)
            })?;
            let k = kinds(&g.nodes()?);
            let want_kernels = piece
                .launches(l)
                .ok_or_else(|| format!("layer {l} is outside the piece"))?;
            // The piece copies nothing: its handoff launch writes the page.
            let want = [want_kernels, 0, 2, 0];
            let mut replay_eq = true;
            for si in (0..sets.len()).rev() {
                inject(stream, &mut io, &dumps[si])?;
                g.launch(stream)?;
                hybrid.serve_captured()?;
                stream.synchronize()?;
                let r = read_piece(&gpu, &piece, &hybrid, &io, &lc)?;
                replay_eq &= first_diff(&r, &eager[si]) == "-";
            }
            drop(g);
            let nodes_ok = k == want;
            let mut line = format!(
                "graph L={l} nodes={} kernels={} memcpy={} memops={} other={} (want {}/{}/{}/{}) \
                 replays_eq_eager={replay_eq}",
                k.iter().sum::<usize>(),
                k[0],
                k[1],
                k[2],
                k[3],
                want[0],
                want[1],
                want[2],
                want[3]
            );
            let mut pass = nodes_ok && replay_eq;
            if !overlap_done && lc.n_card > 0 && fold {
                overlap_done = true;
                let mut off_eq = true;
                for (si, d) in dumps.iter().enumerate() {
                    inject(stream, &mut io, d)?;
                    let r = run_piece(&gpu, &mut piece, &mut hybrid_off, &mut io, &lc)?;
                    off_eq &= first_diff(&r, &eager[si]) == "-";
                }
                line.push_str(&format!(" overlap_off_eq={off_eq}"));
                pass &= off_eq;
            }
            println!("{line} {}", verdict(pass));
            tally.add(pass);
        }
        let stats = hybrid.stats();
        let pass = tally.failed == 0;
        println!(
            "gate_deepseek41_chain_ffn: {} checks across {} sets and {} layers, {} failed, {} \
             waived (near tie) — host served {} layers, {} slots — {}",
            tally.checks,
            sets.len(),
            hp.n_layer,
            tally.failed,
            tally.waived,
            stats.served,
            stats.host_slots,
            verdict(pass)
        );
        if !pass {
            return Err(checks_failed());
        }
        Ok(())
    }
}
