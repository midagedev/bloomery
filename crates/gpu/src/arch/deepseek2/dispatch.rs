//! Launch order: the chain, one layer, and the layer's attention and FFN
//! halves, enqueued asynchronously against the resident arena.

use super::names::LayerNames;
use super::scratch::{Gather, LayerScratch, MoeDims, MoeScratch};
use crate::flash::{
    FlashGeom, FlashInputs, FlashLatentArgs, FlashLatentQ8Args, FlashMerge2Q8Args, FlashMergeArgs,
    FlashMergeQ8Args, FlashSegArgs, FlashSegTwiceArgs,
};
use crate::head::Head;
use crate::model::kernels::{
    HeadsGeom, Q3kGemvHeadsArgs, Q3kGemvHeadsPairArgs, Q8_0GemvHeadsArgs, StepKernels,
};
use crate::model::lookup::{dev_weight, f32_gain, f32_tensor, kq_weight, q8_derived};
use crate::model::probe::{
    Bytes, Observer, StepProbe, act_write_bytes, blocks32_bytes, bsum, gemv_act_bytes, tick,
    weight_bytes,
};
use crate::tensor::{DeviceTensor, Q8Act};
use crate::weights::Weights;
use crate::{Gpu, GpuError, launch_u32};
use cuda_core::DeviceBuffer;
use model::attn::MlaParams;

/// Enqueue the whole decode chain at m = 1: layer 0 with its embedding,
/// every later layer reading the previous layer's output, then the head.
/// Asynchronous throughout — no allocation, no synchronization, no host
/// round trip — so this is both the eager body and what the capture records.
///
/// The residual chain is a copy, not an alias: a layer reads its input from
/// `s.x` and writes its output to `s.l_out` (the down store folds the
/// residual in), and one arena serves every layer, so the boundary is one
/// 8 KiB device-to-device copy per layer — a memcpy node inside the capture.
/// The last layer copies into the head's own input buffer instead.
#[allow(clippy::too_many_arguments)]
pub(super) fn enqueue_chain(
    gpu: &Gpu,
    step: &StepKernels,
    w: &Weights,
    names: &[LayerNames],
    kv: &mut [DeviceTensor<u16>],
    s: &mut LayerScratch,
    mla: &MlaParams,
    moe: Option<&MoeDims>,
    head: &mut Head,
) -> Result<(), GpuError> {
    if names.is_empty() || names.len() != kv.len() {
        return Err(GpuError::shape(
            "enqueue_chain",
            format!("{} layers and {} caches", names.len(), kv.len()),
        ));
    }
    let stream = gpu.stream();
    let last = names.len() - 1;
    for slot in 0..names.len() {
        enqueue_layer(
            gpu,
            step,
            w,
            &names[slot],
            &mut kv[slot],
            s,
            mla,
            moe,
            slot == 0,
            &mut |_, _, _| Ok(()),
        )?;
        if slot == last {
            head.input_mut().copy_from_device_async(&s.l_out, stream)?;
        } else {
            s.x.copy_from_device_async(&s.l_out, stream)?;
        }
    }
    head.enqueue(gpu, w)
}

/// Enqueue one layer's whole step at m = 1: the token embedding when the
/// layer is the first of the model, the attention half every layer shares,
/// and the FFN half the layer's own weights select — the fused dense FFN
/// when the layer has no router, the routed MoE half when it has one.
/// Mirrors `model::attn::block_attn_cached` and `model::moe`'s op order with
/// the gated kernels plus this file's gather. Asynchronous throughout —
/// capturable as a body. A layer that does not embed reads its input
/// residual from the resident input buffer.
#[allow(clippy::too_many_arguments)]
pub(super) fn enqueue_layer(
    gpu: &Gpu,
    step: &StepKernels,
    w: &Weights,
    names: &LayerNames,
    kv_l: &mut DeviceTensor<u16>,
    s: &mut LayerScratch,
    mla: &MlaParams,
    moe: Option<&MoeDims>,
    embed: bool,
    obs: &mut Observer<'_>,
) -> Result<(), GpuError> {
    let mut i = 0usize;
    if embed {
        // 0. embed(token) — the block input x, kept intact for both residuals.
        gpu.elem().enqueue_embed_rows(
            gpu.stream(),
            kq_weight(w, "token_embd.weight")?,
            &s.token_buf,
            &mut s.x,
        )?;
        // One table row dequantized into x, the id read from its buffer.
        let embd = dev_weight(w, "token_embd.weight")?;
        tick(
            &mut i,
            obs,
            "embed",
            bsum(&[weight_bytes(embd, 1), Some(4), Some(4 * s.x.len())]),
        )?;
    }
    enqueue_attn(gpu, step, w, names, kv_l, s, mla, &mut i, obs)?;
    if names.routed {
        let dims = moe.ok_or_else(|| {
            GpuError::shape(
                "enqueue_layer",
                format!(
                    "layer {} routes but the stage carries no MoE shapes",
                    names.layer
                ),
            )
        })?;
        enqueue_ffn_moe(gpu, w, names, s, mla, dims, &mut i, obs)
    } else {
        enqueue_ffn_dense(gpu, w, names, s, mla, &mut i, obs)
    }
}

/// Enqueue the attention half of layer `names.layer`: `s.x` (the layer's
/// input residual) in, `s.ffn_inp` (that residual plus the attention output)
/// out, appending the step's key row to `kv_l` and attending over the live
/// rows. Every layer runs this same chain against its own weights and its
/// own cache.
#[allow(clippy::too_many_arguments)]
fn enqueue_attn(
    gpu: &Gpu,
    step: &StepKernels,
    w: &Weights,
    names: &LayerNames,
    kv_l: &mut DeviceTensor<u16>,
    s: &mut LayerScratch,
    mla: &MlaParams,
    i: &mut usize,
    obs: &mut Observer<'_>,
) -> Result<(), GpuError> {
    attn_proj(gpu, w, names, kv_l, s, mla, i, obs)?;
    attn_qrows(gpu, step, w, names, s, mla, i, obs)?;
    // The last launch of either flash path also writes `kqvc`'s q8_1 form:
    // a block there holds one head's whole latent row, which is four whole
    // q8_1 blocks, so the quantization rides inside it instead of taking a
    // launch of its own. Both paths carry it or neither does — a half-done
    // fold would leave the single-segment path unquantized. The kqvc
    // quantizer reads the same flag to skip its own launch.
    let fold_quant = !s.probe_cfg.skip_quant && !s.probe_cfg.split_flash_quant;
    attn_flash(gpu, kv_l, s, mla, fold_quant, i, obs)?;
    attn_kqvc_quant(gpu, s, mla, fold_quant, i, obs)?;
    attn_wv_b(gpu, step, w, names, s, mla, i, obs)?;
    attn_out(gpu, w, names, s, i, obs)
}

/// Stages 1–5 of the attention half: the norm, the q and kv_a projections,
/// the rope of q, and the fused key path that appends this step's row to
/// `kv_l`.
#[allow(
    clippy::too_many_arguments,
    reason = "a stage of `enqueue_attn`, taking its caller's arguments; the context struct is its own round (rust-quality R8)"
)]
fn attn_proj(
    gpu: &Gpu,
    w: &Weights,
    names: &LayerNames,
    kv_l: &mut DeviceTensor<u16>,
    s: &mut LayerScratch,
    mla: &MlaParams,
    i: &mut usize,
    obs: &mut Observer<'_>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let (hidden, latent, rope) = (s.dims.hidden, mla.latent, mla.rope_dims);
    let kv_width = latent + rope;

    // 1. attn_norm(x) — the dump's FUSED_RMS_NORM output — in both forms the
    //    chain needs: the f32 vector (the block's `attn_norm` tap) and the
    //    q8_1 activation the two projections share, from one launch.
    gpu.fused().enqueue_norm_quant(
        stream,
        &s.x,
        f32_gain(w, &names.attn_norm)?,
        mla.eps,
        &mut s.act_q,
        &mut s.normed,
    )?;
    // x and the gain in, the five q8_1 planes and the f32 vector out.
    tick(
        i,
        obs,
        "attn_norm_quant",
        bsum(&[
            Some(8 * hidden),
            Some(act_write_bytes(&s.act_q, 1)),
            Some(4 * hidden),
        ]),
    )?;
    // 2. the two projections over that one q8_1 activation.
    gpu.enqueue_gemv_q3k(kq_weight(w, &names.attn_q)?, &s.act_q, &mut s.q)?;
    let wq = dev_weight(w, &names.attn_q)?;
    tick(
        i,
        obs,
        "gemv_q3k(attn_q)",
        bsum(&[
            weight_bytes(wq, wq.rows()),
            gemv_act_bytes(wq, &s.act_q, 1),
            Some(4 * s.q.len()),
        ]),
    )?;
    gpu.enqueue_gemv_q3k(kq_weight(w, &names.attn_kv_a_mqa)?, &s.act_q, &mut s.kv_a)?;
    let wkva = dev_weight(w, &names.attn_kv_a_mqa)?;
    tick(
        i,
        obs,
        "gemv_q3k(attn_kv_a_mqa)",
        bsum(&[
            weight_bytes(wkva, wkva.rows()),
            gemv_act_bytes(wkva, &s.act_q, 1),
            Some(4 * s.kv_a.len()),
        ]),
    )?;
    // 4. rope over every 64-value column of both projections: the q layout
    //    puts each head's rope slice on a column boundary (3 columns per
    //    head, the slice last), the kv layout its rope tail (last column);
    //    the rotated neighbours are never read.
    gpu.elem().enqueue_rope(
        stream,
        &s.q,
        &s.cs_buf,
        rope,
        launch_u32("enqueue_attn", "q_cols", s.dims.q_cols)?,
        1,
        &mut s.q_rope_all,
    )?;
    // Every value of q rotated against the position's cos/sin pair.
    tick(
        i,
        obs,
        "rope(q)",
        bsum(&[
            Some(4 * s.q.len()),
            Some(4 * rope),
            Some(4 * s.q_rope_all.len()),
        ]),
    )?;
    // 5. the whole key path of this step in one launch: the latent norm and
    //    the rope of the key's tail leave `kv_s` as
    //    `[kv_compressed | k_rope]`, the permutation leaves `kvr` as the
    //    oracle's CONCAT order `[k_rope | kv_compressed]`, and that row is
    //    appended to the cache as f16 at `pos_buf`'s position — so a
    //    captured replay follows the position. The append moves ahead of the
    //    q legs below; nothing between them reads the cache.
    gpu.fused().enqueue_kv_norm_rope_append(
        stream,
        &s.kv_a,
        f32_gain(w, &names.attn_kv_a_norm)?,
        &s.cs_buf,
        &s.pos_buf,
        mla.eps,
        latent,
        rope,
        &mut s.kv_s,
        &mut s.kvr,
        kv_l,
    )?;
    // kv_a, the latent gain and the cos/sin pair in; the two f32 forms and
    // the one f16 cache row out.
    tick(
        i,
        obs,
        "kv_norm_rope_append",
        bsum(&[
            Some(4 * (kv_width + latent + rope + 1)),
            Some(4 * (s.kv_s.len() + s.kvr.len())),
            Some(2 * kv_width),
        ]),
    )?;
    Ok(())
}

/// Stages 6–7 of the attention half: the flash query rows
/// `[q_rope | q_nope2]` per head, assembled in `s.f_rows`.
#[allow(
    clippy::too_many_arguments,
    reason = "a stage of `enqueue_attn`, taking its caller's arguments; the context struct is its own round (rust-quality R8)"
)]
fn attn_qrows(
    gpu: &Gpu,
    step: &StepKernels,
    w: &Weights,
    names: &LayerNames,
    s: &mut LayerScratch,
    mla: &MlaParams,
    i: &mut usize,
    obs: &mut Observer<'_>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let (latent, rope) = (mla.latent, mla.rope_dims);
    let kv_width = latent + rope;
    let gather =
        |gt: &Gather, src: &DeviceBuffer<f32>, y: &mut DeviceBuffer<f32>| -> Result<(), GpuError> {
            step.enqueue_gather(stream, src, &gt.src, &gt.dst, gt.n, y)
        };

    // 6. q_nope2 per head: one per-head launch dots every derived wk_b row
    //    of head h against q's nope slice of that head (x base h*kq_head,
    //    m = 1 per row), writing the nope2 span of flash row h directly
    //    (y base h*kv_width + rope).
    let (qn2_qs, qn2_d) = q8_derived(w, &names.derived)?;
    step.enqueue_q8_0_gemv_heads(
        stream,
        Q8_0GemvHeadsArgs {
            qs: qn2_qs,
            d: qn2_d,
            x: &s.q,
            rows_per_head: latent,
            x_head_stride: mla.kq_head,
            y_head_stride: kv_width,
            y_off: rope,
            y: &mut s.f_rows,
        },
    )?;
    // Every derived row, each head's nope slice of q, the nope2 spans out.
    let wqn2 = dev_weight(w, &names.derived)?;
    tick(
        i,
        obs,
        "gemv_q8_0_heads(q_nope2)",
        bsum(&[
            weight_bytes(wqn2, wqn2.rows()),
            Some(4 * mla.n_head * mla.nope),
            Some(4 * mla.n_head * latent),
        ]),
    )?;
    // 7. the flash q rows `[q_rope | q_nope2]` per head — the rope spans are
    //    copies of q_rope_all's per-head rope slices (the nope2 span is
    //    already in place).
    gather(&s.g_f_rope_lo, &s.q_rope_all, &mut s.f_rows)?;
    // Per pair: two index words, one value read, one value written.
    tick(
        i,
        obs,
        "gather(f_rope_lo)",
        bsum(&[Some(16 * s.g_f_rope_lo.n)]),
    )?;
    gather(&s.g_f_rope_hi, &s.q_rope_all, &mut s.f_rows)?;
    tick(
        i,
        obs,
        "gather(f_rope_hi)",
        bsum(&[Some(16 * s.g_f_rope_hi.n)]),
    )?;
    Ok(())
}

/// Stage 8 of the attention half: flash over the live rows of `kv_l` into
/// `s.kqvc`, and its q8_1 form alongside when `fold_quant` is set.
fn attn_flash(
    gpu: &Gpu,
    kv_l: &DeviceTensor<u16>,
    s: &mut LayerScratch,
    mla: &MlaParams,
    fold_quant: bool,
    i: &mut usize,
    obs: &mut Observer<'_>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let (latent, rope) = (mla.latent, mla.rope_dims);
    let kv_width = latent + rope;
    // Live key rows of this step, from the host mirror of `pos_buf`: the
    // count the flash kernels read lives on the device, so the byte
    // accounting takes it from the refresh that wrote it.
    let n_keys = s.pos_host as usize + 1;

    // 8. attend over `n_keys` rows — the count lives in a device buffer, so
    //    a captured replay follows it.
    // The merge pass exists only when the cache is tall enough to be cut
    // into segments; a one-segment cache runs the single-block kernel and
    // this layer is one node shorter. The choice is the cache height's, so
    // it cannot differ between capture and replay. The two launches are
    // enqueued separately so an observer times each on its own.
    // The side output's bytes: the same activation planes the standalone
    // quantizer wrote.
    let side_bytes = act_write_bytes(&s.act_kv_lo, s.act_kv_lo.m())
        + act_write_bytes(&s.act_kv_hi, s.act_kv_hi.m());
    let inputs = FlashInputs {
        q: &s.f_rows,
        kv: kv_l,
        n_keys_buf: &s.n_keys_buf,
        kq_scale: mla.kq_scale,
        geom: FlashGeom {
            tokens: 1,
            heads: mla.n_head,
            rope_dims: rope,
            latent_dims: latent,
        },
    };
    if crate::flash::segments_for(kv_l.rows()) == 1 {
        if fold_quant {
            gpu.flash().enqueue_flash_latent_q8(
                stream,
                FlashLatentQ8Args {
                    inputs,
                    y: &mut s.kqvc,
                    lo: &mut s.act_kv_lo,
                    hi: &mut s.act_kv_hi,
                },
            )?;
        } else {
            gpu.flash().enqueue_flash_latent(
                stream,
                FlashLatentArgs {
                    inputs,
                    y: &mut s.kqvc,
                },
            )?;
        }
        // The query rows, the live cache rows as f16, the attended output,
        // and the quantized form when it rides along.
        tick(
            i,
            obs,
            "flash_latent",
            bsum(&[
                Some(4 * (mla.n_head * kv_width + 1)),
                Some(2 * n_keys * kv_width),
                Some(4 * mla.n_head * latent),
                Some(if fold_quant { side_bytes } else { 0 }),
            ]),
        )?;
    } else {
        // A flash lever swaps the segment launch for the probe entry that
        // does one stage twice — the same launch, the same partials. The
        // tensor-core pass is the third shape of that one launch: one block
        // per (head group, segment) instead of per (head, segment), the
        // same partials, so the merge below and the launch count do not
        // move.
        let seg = FlashSegArgs {
            inputs,
            part_v: &mut s.part_v,
            part_ms: &mut s.part_ms,
        };
        match s.probe_cfg.flash_seg_twice() {
            None if crate::flash::flash_mma() => {
                gpu.flash().enqueue_flash_latent_mma(stream, seg)?
            }
            None => gpu.flash().enqueue_flash_latent_seg(stream, seg)?,
            Some((twice, shift)) => gpu.flash().enqueue_flash_latent_seg_twice(
                stream,
                FlashSegTwiceArgs {
                    seg,
                    twice,
                    shift_rows: shift,
                },
            )?,
        }
        // Same reads as the single-block launch; the partials of the live
        // segments out, plus the `(−inf, 0)` pair every segment past the
        // live keys still writes so the merge can skip it.
        let segs = crate::flash::segments_for(kv_l.rows());
        let live = n_keys.div_ceil(crate::flash::seg_keys()).min(segs);
        tick(
            i,
            obs,
            "flash_latent",
            bsum(&[
                Some(4 * (mla.n_head * kv_width + 1)),
                Some(2 * n_keys * kv_width),
                Some(4 * mla.n_head * live * latent),
                Some(8 * mla.n_head * segs),
            ]),
        )?;
        let merge = FlashMergeArgs {
            n_keys_buf: &s.n_keys_buf,
            cache_rows: kv_l.rows(),
            tokens: 1,
            heads: mla.n_head,
            latent_dims: latent,
            part_v: &s.part_v,
            part_ms: &s.part_ms,
            y: &mut s.kqvc,
        };
        if fold_quant {
            if s.probe_cfg.flash_merge2 {
                gpu.flash().enqueue_flash_merge2_q8(
                    stream,
                    FlashMerge2Q8Args {
                        merge,
                        lo: &mut s.act_kv_lo,
                        hi: &mut s.act_kv_hi,
                        shift_segs: 0,
                    },
                )?;
            } else {
                gpu.flash().enqueue_flash_merge_q8(
                    stream,
                    FlashMergeQ8Args {
                        merge,
                        lo: &mut s.act_kv_lo,
                        hi: &mut s.act_kv_hi,
                    },
                )?;
            }
        } else {
            gpu.flash().enqueue_flash_merge(stream, merge)?;
        }
        // The live segments' partials in, the attended output out, and the
        // quantized form when it rides along.
        tick(
            i,
            obs,
            "flash_merge",
            bsum(&[
                Some(4),
                Some(4 * mla.n_head * live * latent),
                Some(8 * mla.n_head * live),
                Some(4 * mla.n_head * latent),
                Some(if fold_quant { side_bytes } else { 0 }),
            ]),
        )?;
    }
    Ok(())
}

/// Stage 9 of the attention half, first part: `s.kqvc`'s q8_1 form in
/// `s.act_kv_lo`/`s.act_kv_hi`, unless flash already wrote it
/// (`fold_quant`) or the probe skips it.
fn attn_kqvc_quant(
    gpu: &Gpu,
    s: &mut LayerScratch,
    mla: &MlaParams,
    fold_quant: bool,
    i: &mut usize,
    obs: &mut Observer<'_>,
) -> Result<(), GpuError> {
    let latent = mla.latent;
    let half = mla.n_head / 2;

    // 9. wv_b per head: flash's output is head-major — kqvc column h is
    //    head h's compressed values, the m-column layout the quantizer
    //    consumes; the heads 8..15 half quantizes from kqvc's base offset
    //    (the quantizer's x base), so no gathered copy.
    if !fold_quant && !s.probe_cfg.skip_quant {
        if s.probe_cfg.split_kqvc {
            gpu.enqueue_quantize_q8_1(&s.kqvc, &mut s.act_kv_lo)?;
            tick(
                i,
                obs,
                "quantize_q8_1(kqvc_lo)",
                bsum(&[
                    Some(4 * s.act_kv_lo.m() * latent),
                    Some(act_write_bytes(&s.act_kv_lo, s.act_kv_lo.m())),
                ]),
            )?;
            gpu.enqueue_quantize_q8_1_at(&s.kqvc, half * latent, &mut s.act_kv_hi)?;
            tick(
                i,
                obs,
                "quantize_q8_1(kqvc_hi)",
                bsum(&[
                    Some(4 * s.act_kv_hi.m() * latent),
                    Some(act_write_bytes(&s.act_kv_hi, s.act_kv_hi.m())),
                ]),
            )?;
        } else {
            // Both halves of `kqvc` in one launch: the same two planes of
            // the same buffer, one grid covering both.
            let (lo, hi) = (&mut s.act_kv_lo, &mut s.act_kv_hi);
            gpu.enqueue_quantize_q8_1_pair(&s.kqvc, 0, lo, half * latent, hi)?;
            tick(
                i,
                obs,
                "quantize_q8_1(kqvc)",
                bsum(&[
                    Some(4 * (s.act_kv_lo.m() + s.act_kv_hi.m()) * latent),
                    Some(act_write_bytes(&s.act_kv_lo, s.act_kv_lo.m())),
                    Some(act_write_bytes(&s.act_kv_hi, s.act_kv_hi.m())),
                ]),
            )?;
        }
    }
    Ok(())
}

/// Stage 9 of the attention half, second part: wv_b per head over the
/// quantized `kqvc` into `s.kqv_2d`.
#[allow(
    clippy::too_many_arguments,
    reason = "a stage of `enqueue_attn`, taking its caller's arguments; the context struct is its own round (rust-quality R8)"
)]
fn attn_wv_b(
    gpu: &Gpu,
    step: &StepKernels,
    w: &Weights,
    names: &LayerNames,
    s: &mut LayerScratch,
    mla: &MlaParams,
    i: &mut usize,
    obs: &mut Observer<'_>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let half = mla.n_head / 2;

    // Each per-head launch dots only its heads' wv_b rows (absolute row
    // h*(nope+v_head) + nope + j, activation column h - head_base, m = 1),
    // writing the flat, head-major kqv_2d directly — the wk_b rows are
    // never read.
    let kv_b = kq_weight(w, &names.attn_kv_b)?;
    let wkvb = dev_weight(w, &names.attn_kv_b)?;
    let geom = HeadsGeom {
        head_base: 0,
        rows_per_head: mla.v_head,
        row_stride_per_head: mla.nope + mla.v_head,
        row_off: mla.nope,
        y_head_stride: mla.v_head,
    };
    if s.probe_cfg.split_heads {
        // The two-launch form: half the heads' wv_b rows against their own
        // activation columns, then the other half.
        let wv_b_bytes = |a: &Q8Act| -> Bytes {
            bsum(&[
                weight_bytes(wkvb, half * mla.v_head),
                gemv_act_bytes(wkvb, a, a.m()),
                Some(4 * half * mla.v_head),
            ])
        };
        step.enqueue_q3k_gemv_heads(
            stream,
            Q3kGemvHeadsArgs {
                w: kv_b,
                act: &s.act_kv_lo,
                geom,
                y: &mut s.kqv_2d,
            },
        )?;
        tick(i, obs, "gemv_q3k_heads(wv_b_lo)", wv_b_bytes(&s.act_kv_lo))?;
        step.enqueue_q3k_gemv_heads(
            stream,
            Q3kGemvHeadsArgs {
                w: kv_b,
                act: &s.act_kv_hi,
                geom: HeadsGeom {
                    head_base: half,
                    ..geom
                },
                y: &mut s.kqv_2d,
            },
        )?;
        tick(i, obs, "gemv_q3k_heads(wv_b_hi)", wv_b_bytes(&s.act_kv_hi))?;
    } else {
        // Both halves' wv_b rows in one launch, each head against its own
        // activation column. The halves read different weight rows, so this
        // saves the launch, not the weight read.
        step.enqueue_q3k_gemv_heads_pair(
            stream,
            Q3kGemvHeadsPairArgs {
                w: kv_b,
                lo: &s.act_kv_lo,
                hi: &s.act_kv_hi,
                geom,
                y: &mut s.kqv_2d,
            },
        )?;
        tick(
            i,
            obs,
            "gemv_q3k_heads(wv_b)",
            bsum(&[
                weight_bytes(wkvb, mla.n_head * mla.v_head),
                gemv_act_bytes(wkvb, &s.act_kv_lo, s.act_kv_lo.m()),
                gemv_act_bytes(wkvb, &s.act_kv_hi, s.act_kv_hi.m()),
                Some(4 * mla.n_head * mla.v_head),
            ]),
        )?;
    }
    Ok(())
}

/// Stage 10 of the attention half: attn_output over `s.kqv_2d` and the
/// attention residual into `s.ffn_inp`, then the probe's empty nodes when
/// it is armed.
fn attn_out(
    gpu: &Gpu,
    w: &Weights,
    names: &LayerNames,
    s: &mut LayerScratch,
    i: &mut usize,
    obs: &mut Observer<'_>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let hidden = s.dims.hidden;

    // 10. attn_output over the flat kqv_2d, then the attention residual.
    if !s.probe_cfg.skip_quant {
        gpu.enqueue_quantize_q8_1(&s.kqv_2d, &mut s.act_ao)?;
        tick(
            i,
            obs,
            "quantize_q8_1(act_ao)",
            bsum(&[
                Some(4 * s.kqv_2d.len()),
                Some(act_write_bytes(&s.act_ao, s.act_ao.m())),
            ]),
        )?;
    }
    gpu.enqueue_gemv_q4k(
        kq_weight(w, &names.attn_output)?,
        &s.act_ao,
        &mut s.attn_out,
    )?;
    let wao = dev_weight(w, &names.attn_output)?;
    tick(
        i,
        obs,
        "gemv_q4k(attn_output)",
        bsum(&[
            weight_bytes(wao, wao.rows()),
            gemv_act_bytes(wao, &s.act_ao, 1),
            Some(4 * s.attn_out.len()),
        ]),
    )?;
    gpu.elem()
        .enqueue_add(stream, &s.attn_out, &s.x, hidden, &mut s.ffn_inp)?;
    tick(i, obs, "add(attn_resid)", bsum(&[Some(12 * hidden)]))?;
    // The probe's empty nodes, if it is armed. Nothing downstream reads
    // `probe_buf`, so these change the node count and nothing else.
    for _ in 0..s.probe_cfg.pad_per_layer {
        s.probe.enqueue_touch(stream, &mut s.probe_buf)?;
        tick(i, obs, "probe_pad", bsum(&[Some(4 * 32)]))?;
    }
    Ok(())
}

/// Enqueue the dense FFN half: norm+quantize, gate·up·swiglu, 32-value
/// quantize, down+residual — bit-identical to the op path (the P0b
/// contract). `s.ffn_inp` in, `s.l_out` out.
#[allow(clippy::too_many_arguments)]
fn enqueue_ffn_dense(
    gpu: &Gpu,
    w: &Weights,
    names: &LayerNames,
    s: &mut LayerScratch,
    mla: &MlaParams,
    i: &mut usize,
    obs: &mut Observer<'_>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let hidden = s.dims.hidden;
    let LayerScratch {
        ffn_inp,
        act_ffn,
        dense,
        l_out,
        ..
    } = s;
    let d = dense.as_mut().ok_or_else(|| {
        GpuError::shape(
            "enqueue_ffn_dense",
            format!(
                "layer {} is dense but the stage carries no dense arena",
                names.layer
            ),
        )
    })?;
    // The dense half has no f32 consumer of the norm: `h` takes the side
    // output and the next launch overwrites it.
    gpu.fused().enqueue_norm_quant(
        stream,
        ffn_inp,
        f32_gain(w, &names.ffn_norm)?,
        mla.eps,
        act_ffn,
        &mut d.h,
    )?;
    tick(
        i,
        obs,
        "ffn_norm_quant",
        bsum(&[
            Some(8 * hidden),
            Some(act_write_bytes(act_ffn, 1)),
            Some(4 * hidden),
        ]),
    )?;
    gpu.fused().enqueue_gate_up_swiglu(
        stream,
        kq_weight(w, &names.ffn_gate)?,
        kq_weight(w, &names.ffn_up)?,
        act_ffn,
        &mut d.h,
    )?;
    // Both projections' rows over the one activation, the swiglu out.
    let (wg, wu) = (
        dev_weight(w, &names.ffn_gate)?,
        dev_weight(w, &names.ffn_up)?,
    );
    tick(
        i,
        obs,
        "ffn_gate_up_swiglu",
        bsum(&[
            weight_bytes(wg, wg.rows()),
            weight_bytes(wu, wu.rows()),
            gemv_act_bytes(wg, act_ffn, 1),
            Some(4 * wg.rows()),
        ]),
    )?;
    gpu.q5().enqueue_quantize_q8(stream, &d.h, &mut d.act32)?;
    tick(
        i,
        obs,
        "ffn_quantize_q8",
        bsum(&[Some(4 * d.h.len()), Some(blocks32_bytes(&d.act32, 1))]),
    )?;
    gpu.fused().enqueue_down_add_q5_1(
        stream,
        kq_weight(w, &names.ffn_down)?,
        &d.act32,
        ffn_inp,
        l_out,
    )?;
    // Every down row, the quantized swiglu, the residual, the block output.
    let wd = dev_weight(w, &names.ffn_down)?;
    tick(
        i,
        obs,
        "ffn_down_add",
        bsum(&[
            weight_bytes(wd, wd.rows()),
            Some(blocks32_bytes(&d.act32, 1)),
            Some(8 * hidden),
        ]),
    )?;
    Ok(())
}

/// Enqueue the routed MoE FFN half at m = 1: `s.ffn_inp` in, `s.l_out` out.
///
/// The norm's launch takes the f32 side output here, unlike the dense
/// half's: the router eats the f32 normed vector and the experts eat its
/// q8_1 form, so both must exist. The routed experts run through the
/// device-resident `sel` (the router's own ids buffer at m = 1), so a
/// captured graph follows the routing; the shared expert runs the dense
/// fused kernels at its own width on the same quantized input. The combine
/// keeps the dump's grouping — `(Σ w·down + shexp) + resid`.
#[allow(clippy::too_many_arguments)]
fn enqueue_ffn_moe(
    gpu: &Gpu,
    w: &Weights,
    names: &LayerNames,
    s: &mut LayerScratch,
    mla: &MlaParams,
    dims: &MoeDims,
    i: &mut usize,
    obs: &mut Observer<'_>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let hidden = s.dims.hidden;
    let LayerScratch {
        ffn_inp,
        act_ffn,
        moe,
        l_out,
        probe_cfg:
            StepProbe {
                skip_quant,
                split_moe_quant,
                ..
            },
        ..
    } = s;
    let m = moe.as_mut().ok_or_else(|| {
        GpuError::shape(
            "enqueue_ffn_moe",
            format!(
                "layer {} routes but the stage carries no MoE arena",
                names.layer
            ),
        )
    })?;
    // 1. ffn_norm in both forms the half's consumers need, from one launch:
    //    the router eats the f32 normed vector and the experts its q8_1
    //    form.
    gpu.fused().enqueue_norm_quant(
        stream,
        ffn_inp,
        f32_gain(w, &names.ffn_norm)?,
        mla.eps,
        act_ffn,
        &mut m.normed,
    )?;
    tick(
        i,
        obs,
        "moe_ffn_norm_quant",
        bsum(&[
            Some(8 * hidden),
            Some(act_write_bytes(act_ffn, 1)),
            Some(4 * hidden),
        ]),
    )?;
    moe_route(gpu, w, names, act_ffn, m, dims, hidden, i, obs)?;
    moe_shexp_gate_up(gpu, w, names, act_ffn, m, dims, i, obs)?;
    moe_quantize(gpu, m, dims, *skip_quant, *split_moe_quant, i, obs)?;
    moe_down_combine(gpu, w, names, ffn_inp, l_out, m, dims, hidden, i, obs)
}

/// Stages 2–6 of the routed FFN half: the router over `m.normed`, then the
/// selected experts' fused gate·up·swiglu into `m.h_exp`.
#[allow(
    clippy::too_many_arguments,
    reason = "a stage of `enqueue_ffn_moe`, taking its caller's arguments; the context struct is its own round (rust-quality R8)"
)]
fn moe_route(
    gpu: &Gpu,
    w: &Weights,
    names: &LayerNames,
    act_ffn: &Q8Act,
    m: &mut MoeScratch,
    dims: &MoeDims,
    hidden: usize,
    i: &mut usize,
    obs: &mut Observer<'_>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    // 2-3. the router: an f32 gemv over the normed vector, then softmax +
    //      top-k + scale. `m.ids` is the `sel` every expert launch reads.
    gpu.q8f32().enqueue_f32_gemv(
        stream,
        f32_tensor(w, &names.ffn_gate_inp)?,
        &m.normed,
        1,
        &mut m.logits,
    )?;
    let wr = dev_weight(w, &names.ffn_gate_inp)?;
    tick(
        i,
        obs,
        "moe_router_gemv",
        bsum(&[
            weight_bytes(wr, wr.rows()),
            Some(4 * hidden),
            Some(4 * dims.n_expert),
        ]),
    )?;
    gpu.router().enqueue_router_topk(
        stream,
        &m.logits,
        1,
        dims.scale,
        &mut m.probs,
        &mut m.ids,
        &mut m.weights,
    )?;
    // The logits in; the probabilities, the chosen ids and their weights out.
    tick(
        i,
        obs,
        "moe_router_topk",
        bsum(&[Some(8 * dims.n_expert), Some(8 * dims.n_used)]),
    )?;
    // 4-6. the routed experts, all six per launch through `sel`.
    gpu.moe_fused().enqueue_expert_gate_up_swiglu(
        stream,
        kq_weight(w, &names.ffn_gate_exps)?,
        kq_weight(w, &names.ffn_up_exps)?,
        act_ffn,
        &m.ids,
        dims.n_used,
        dims.ff,
        &mut m.h_exp,
    )?;
    // Only the selected experts' rows are read — `n_used` blocks of
    // `ff` rows out of the stack, the same count for any selection
    // because the ids the router writes are distinct
    // (`profile_layer` asserts that on the ids it reads back).
    let (wge, wue) = (
        dev_weight(w, &names.ffn_gate_exps)?,
        dev_weight(w, &names.ffn_up_exps)?,
    );
    tick(
        i,
        obs,
        "moe_expert_gate_up_swiglu",
        bsum(&[
            weight_bytes(wge, dims.n_used * dims.ff),
            weight_bytes(wue, dims.n_used * dims.ff),
            gemv_act_bytes(wge, act_ffn, 1),
            Some(4 * dims.n_used),
            Some(4 * dims.n_used * dims.ff),
        ]),
    )?;
    Ok(())
}

/// Stage 7 of the routed FFN half: the shared expert's gate·up·swiglu into
/// `m.h_sh`.
#[allow(
    clippy::too_many_arguments,
    reason = "a stage of `enqueue_ffn_moe`, taking its caller's arguments; the context struct is its own round (rust-quality R8)"
)]
fn moe_shexp_gate_up(
    gpu: &Gpu,
    w: &Weights,
    names: &LayerNames,
    act_ffn: &Q8Act,
    m: &mut MoeScratch,
    dims: &MoeDims,
    i: &mut usize,
    obs: &mut Observer<'_>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    // 7. the shared expert's dense fused gate·up·swiglu at its own width, on
    //    the same quantized input. It comes before the quantize below rather
    //    than after the routed experts' down projection: it reads only
    //    `act_ffn` and writes only `h_sh`, so neither its inputs nor its
    //    output meet anything enqueued between here and where it used to
    //    stand — and standing here makes the two quantizations adjacent, so
    //    one launch with two geometries can do both.
    gpu.fused().enqueue_gate_up_swiglu(
        stream,
        kq_weight(w, &names.ffn_gate_shexp)?,
        kq_weight(w, &names.ffn_up_shexp)?,
        act_ffn,
        &mut m.h_sh,
    )?;
    let (wgs, wus) = (
        dev_weight(w, &names.ffn_gate_shexp)?,
        dev_weight(w, &names.ffn_up_shexp)?,
    );
    tick(
        i,
        obs,
        "shexp_gate_up_swiglu",
        bsum(&[
            weight_bytes(wgs, wgs.rows()),
            weight_bytes(wus, wus.rows()),
            gemv_act_bytes(wgs, act_ffn, 1),
            Some(4 * dims.shexp_ff),
        ]),
    )?;
    Ok(())
}

/// Stage 8 of the routed FFN half: `m.h_exp`'s 32-value form and `m.h_sh`'s
/// q8_1 form, in one launch or two as the probe asks, or none when it
/// skips them.
fn moe_quantize(
    gpu: &Gpu,
    m: &mut MoeScratch,
    dims: &MoeDims,
    skip_quant: bool,
    split_moe_quant: bool,
    i: &mut usize,
    obs: &mut Observer<'_>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    // 8. both quantizations in one launch: the routed experts' 32-value form
    //    and the shared expert's q8_1. Different sources, different outputs,
    //    different geometries — merged because they are two launches, not
    //    because they share work.
    if !skip_quant {
        if split_moe_quant {
            gpu.q5()
                .enqueue_quantize_q8(stream, &m.h_exp, &mut m.act32_exp)?;
            tick(
                i,
                obs,
                "moe_expert_quantize_q8",
                bsum(&[
                    Some(4 * dims.n_used * dims.ff),
                    Some(blocks32_bytes(&m.act32_exp, dims.n_used)),
                ]),
            )?;
            gpu.enqueue_quantize_q8_1(&m.h_sh, &mut m.act_sh)?;
            tick(
                i,
                obs,
                "shexp_quantize_q8_1",
                bsum(&[
                    Some(4 * dims.shexp_ff),
                    Some(act_write_bytes(&m.act_sh, m.act_sh.m())),
                ]),
            )?;
        } else {
            gpu.q5().enqueue_quantize_q8_pair(
                stream,
                &m.h_exp,
                &mut m.act32_exp,
                &m.h_sh,
                &mut m.act_sh,
            )?;
            tick(
                i,
                obs,
                "moe_quantize_pair",
                bsum(&[
                    Some(4 * dims.n_used * dims.ff),
                    Some(blocks32_bytes(&m.act32_exp, dims.n_used)),
                    Some(4 * dims.shexp_ff),
                    Some(act_write_bytes(&m.act_sh, m.act_sh.m())),
                ]),
            )?;
        }
    }
    Ok(())
}

/// Stages 9–10 of the routed FFN half: the selected experts' and the shared
/// expert's down projections, then the combine with the residual
/// `ffn_inp` into `l_out`.
#[allow(
    clippy::too_many_arguments,
    reason = "a stage of `enqueue_ffn_moe`, taking its caller's arguments; the context struct is its own round (rust-quality R8)"
)]
fn moe_down_combine(
    gpu: &Gpu,
    w: &Weights,
    names: &LayerNames,
    ffn_inp: &DeviceBuffer<f32>,
    l_out: &mut DeviceBuffer<f32>,
    m: &mut MoeScratch,
    dims: &MoeDims,
    hidden: usize,
    i: &mut usize,
    obs: &mut Observer<'_>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    gpu.q5().enqueue_gemv_q5_0_sel(
        stream,
        kq_weight(w, &names.ffn_down_exps)?,
        &m.act32_exp,
        &m.ids,
        dims.n_used,
        hidden,
        &mut m.down,
    )?;
    let wde = dev_weight(w, &names.ffn_down_exps)?;
    tick(
        i,
        obs,
        "moe_expert_down",
        bsum(&[
            weight_bytes(wde, dims.n_used * hidden),
            Some(blocks32_bytes(&m.act32_exp, dims.n_used)),
            Some(4 * dims.n_used),
            Some(4 * dims.n_used * hidden),
        ]),
    )?;
    // 9. the shared expert's Q4_K down projection, whose K is the shared
    //    width (odd super-block count, which the Q4_K row geometry takes).
    gpu.enqueue_gemv_q4k(
        kq_weight(w, &names.ffn_down_shexp)?,
        &m.act_sh,
        &mut m.shexp,
    )?;
    let wds = dev_weight(w, &names.ffn_down_shexp)?;
    tick(
        i,
        obs,
        "shexp_down",
        bsum(&[
            weight_bytes(wds, wds.rows()),
            gemv_act_bytes(wds, &m.act_sh, 1),
            Some(4 * hidden),
        ]),
    )?;
    // 10. (Σ w·down + shexp) + resid, the dump's own grouping.
    gpu.moe_fused().enqueue_moe_combine(
        stream,
        &m.down,
        &m.weights,
        &m.shexp,
        ffn_inp,
        hidden,
        dims.n_used,
        l_out,
    )?;
    // The slots' down projections and weights, the shared expert, the
    // residual, the block output.
    tick(
        i,
        obs,
        "moe_combine",
        bsum(&[Some(4 * dims.n_used * (hidden + 1)), Some(12 * hidden)]),
    )?;
    Ok(())
}
