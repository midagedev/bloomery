//! GPU gate for the Pixtral image tower on the card (`bloomery_gpu_vision::pixtral`, the `pixtral`
//! projector of Mistral Large 4): the whole chain from an image's patches to the rows the text model
//! reads, against llama.cpp mainline's mtmd graph (the sets of `just dump-ref-ml4vis`: set A the
//! preprocess, set B the last block's `layer_out` and `embd`), and every kernel of block 0 and of the
//! projector against its host rule.
//!
//! The oracle is a CPU build, so a set has no card run and no `.cpu` twin: a tap's controls are the
//! reference rounded to bf16 against itself (the floor of any bf16 tower) and the network's
//! sensitivity — our own encode against our encode with the last bit of one patch value in sixteen
//! flipped. The reference's ViT rows are the patches in raster order and ours in merge order, so the
//! ViT's output is mapped to our order before it is compared ([`to_merge`]); the rows the text model
//! reads (tokens and `[IMG_BREAK]` rows) are in the same order on both sides.
//!
//! Per image of set B, in chain order (the first failing line names the earliest op):
//! 1. The host half: the plan and the patches of `vision::arch::pixtral::preprocess` against set A's
//!    `# image` line and `inp_raw`, bit for bit after the bf16 rounding, and the image's rows
//!    (`nx·ny + ny − 1`) against the set's token count.
//! 2. The ViT's output (`layer_out` of the last block) and the rows (`embd`) against the reference's
//!    f32, as `rms(ours − ref) / rms(ref)`, within [`K_TAP`] times the larger of its controls.
//! 3. The projector teacher-forced from the reference's ViT output; the band is [`K_TAP`] times the
//!    larger of the rounding floor and the step's own sensitivity.
//! 4. The rows: each grid row's tokens are the projector's rows and the row after each grid row but
//!    the last is the file's `[IMG_BREAK]` row, bit for bit.
//! 5. On the set's image that is no resize (the `FULL` image), block 0 and the projector op by op
//!    against the host rules of `bloomery_gpu_vision`, each on our own tapped inputs: the norms and
//!    the epilogues bit for bit; every GEMM inside its accumulation-order bound, the query, key and
//!    value against the file's own matrices (the load's permutation of the query and key rows undone)
//!    and the merger against the file's matrix in `im2col`'s order (the load's reorder not used); the
//!    RoPE against llama.cpp's `build_rope_2d` transcribed from its source, in the file's order of a
//!    head; the attention (heads of 128) inside its f32 bound; the SiLU gate and the tanh GELU
//!    against their host rules.
//! 6. Launches per image, two plain encodes bit-identical to each other and to the tapped run.
//!
//! Then once: an encode does not depend on the larger image encoded before it in the reused
//! buffers; the bytes on the card equal the card figure.

#[cfg(feature = "vision")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_pixtral_tower", gate::run())
}

#[cfg(feature = "vision")]
#[path = "shared/vision_rules.rs"]
#[allow(
    dead_code,
    reason = "shared by the tower gates; each runs a part of it"
)]
mod vision_rules;

#[cfg(feature = "vision")]
#[path = "shared/vision_taps.rs"]
#[allow(
    dead_code,
    reason = "shared by the tower gates; each runs a part of it"
)]
mod vision_taps;

#[cfg(feature = "vision")]
mod gate {
    use std::collections::{HashMap, HashSet};

    use super::vision_rules::{Count, exact_check, gemm_check, near_check, par_rows};
    use super::vision_taps::{
        Collect, HostFile, band_check, dist, eager_twice, floor, nudge, nudged, read_image,
        reuse_after_largest, sens as sensitivity, to_bf16, to_f32,
    };
    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use bloomery_gpu_vision::attn::{QkvLayout, attn_bound};
    use bloomery_gpu_vision::attn_h128::{HEAD_DIM_128, attn_128_ref};
    use bloomery_gpu_vision::f32_bf16;
    use bloomery_gpu_vision::gelu_tanh::gelu_tanh_ref;
    use bloomery_gpu_vision::gemm_bf16::{Epilogue, epilogue_ref, within_order};
    use bloomery_gpu_vision::mlp::silu_mul_ref;
    use bloomery_gpu_vision::norm::rms_norm_ref;
    use bloomery_gpu_vision::pixtral::{Encoder, scratch_bytes_of, weight_bytes_of};
    use bloomery_gpu_vision::rope2d::{merge_order, rope_pair_ref, rope_source_128};
    use gguf::Gguf;
    use refset::arch::mistrallarge4::vis::{MMPROJ, PREPROC, TAPS};
    use refset::clefvis::{ClefvisSet, EMBD, PIXTRAL_VIT_OUT};
    use vision::Patches;
    use vision::arch::pixtral::{
        Hparams, ROPE_THETA, TokenLimits, names as n, preprocess, rows_of,
    };

    const NAME: &str = "gate_pixtral_tower";
    const PREPROC_SET: &str = "ref_mistrallarge4_vis_preproc";
    const TAPS_SET: &str = "ref_mistrallarge4_vis_taps";
    /// The set's image whose block 0 and projector the op rules run on: in the budget and a
    /// multiple of 28, so no resize touches it; 16 grid rows, so 15 break rows.
    const FULL: &str = "free-448x448";
    /// The set's image with the largest grid the token limit allows.
    const LARGEST: &str = "max-2600x1800";
    /// The card figures of the file's tower [derived]: the weights from its tensor table
    /// (`card::weights`: the file's 3,444,678,656 B of tensor data less the patch kernel and the
    /// patch merger narrowed from f32 to bf16 and the `[IMG_BREAK]` row), the scratch at the file's
    /// 1,024-token limit (`card::ScratchLens`: 4,096 patches; 330,336 B a token).
    const WEIGHT_BYTES: u64 = 3_408_691_200;
    const SCRATCH_BYTES: u64 = 338_264_064;
    /// Launches of an image of more than one grid row: the patch GEMM and the pre-norm, nine per
    /// block of 30, the projector's five, the copy of the tokens and the copy of the break rows.
    const LAUNCHES: usize = 279;
    /// A tap passes when `rms(ours − ref) / rms(ref)` is at most this multiple of the larger of its
    /// controls.
    /// PIN(2026-11-02): fixed before the first run of this tower against its set: the Qwen3-VL
    /// and RADIO towers' multiple, for the same chain of bf16 ops (30 blocks); the design's
    /// prediction is a block output at about 1.6× the rounding floor, and 4 leaves room for a
    /// free-running network's amplification without admitting an order-one defect.
    const K_TAP: f64 = 4.0;
    /// Values of block 0's SiLU-gated rows of the full-tap image (1,024 × 6,144) that may differ
    /// from the host rule, each by one bf16 step at most: the device's `expf` against the host's.
    /// PIN(2026-11-02): fixed before the first run: the same kernel (`vis_silu_mul`) measured 0 of
    /// 4,283,136 on `gate_vision_encoder`'s full-tap image; this gate's first rule line is the
    /// evidence for it.
    const SILU_ROWS_PIN: usize = 0;
    /// The same for the projector's tanh GELU rows of the full-tap image (256 × 6,144).
    /// PIN(2026-11-02): fixed before the first run: the same kernel (`vis_gelu_tanh`) measured 0 on
    /// the Qwen3-VL tower's block-0 and merger rows; this gate's first rule line is the evidence.
    const GELU_ROWS_PIN: usize = 0;

    // ------------------------------------------------------------ row order

    /// The index, in merge order, of the patch at row `y`, column `x` of a grid `w` patches wide:
    /// the groups of 2×2 patches in raster order, the group's patches in raster order inside it.
    fn merge_index(y: usize, x: usize, w: usize) -> usize {
        ((y / 2) * (w / 2) + x / 2) * 4 + (y % 2) * 2 + x % 2
    }

    /// The oracle's patch rows of width `dim` (raster order) in merge order.
    fn to_merge(refv: &[f32], dim: usize, (h, w): (usize, usize)) -> Result<Vec<f32>, GateError> {
        if refv.len() != h * w * dim {
            return Err(format!("{} values for {h}x{w} patches of {dim}", refv.len()).into());
        }
        let mut out = vec![0.0f32; refv.len()];
        for y in 0..h {
            for x in 0..w {
                let (src, dst) = ((y * w + x) * dim, merge_index(y, x, w) * dim);
                out[dst..dst + dim].copy_from_slice(&refv[src..src + dim]);
            }
        }
        Ok(out)
    }

    /// The `dim` columns at `c0` of every row of `width`.
    fn cols(bits: &[u16], width: usize, c0: usize, dim: usize) -> Vec<u16> {
        bits.chunks_exact(width)
            .flat_map(|r| r[c0..c0 + dim].iter().copied())
            .collect()
    }

    /// A row segment of heads of 128 from the card's order to the file's: the card's value `r` of a
    /// head is the file's value `rope_source_128(r)` of it (the load's permutation of the query
    /// and key rows).
    fn heads_to_file(seg: &[u16]) -> Vec<u16> {
        let mut out = vec![0u16; seg.len()];
        for (h, head) in seg.as_chunks::<HEAD_DIM_128>().0.iter().enumerate() {
            for (r, &v) in head.iter().enumerate() {
                out[h * HEAD_DIM_128 + rope_source_128(r)] = v;
            }
        }
        out
    }

    /// Every `dim`-wide row of `bits` (rows of `width`, the segment at `c0`) in the file's order.
    fn rows_to_file(bits: &[u16], width: usize, c0: usize, dim: usize) -> Vec<u16> {
        bits.chunks_exact(width)
            .flat_map(|r| heads_to_file(&r[c0..c0 + dim]))
            .collect()
    }

    /// llama.cpp's `build_rope_2d` with interleaved frequencies on one head of 128 in the file's
    /// order, transcribed from its source: two NORM-mode `ggml_rope_ext` calls of 64 values, each
    /// turning adjacent pairs `(2i, 2i + 1)`; the first over values 0..64 at the patch's row, the
    /// second over 64..128 at its column with `freq_scale = θ^(−2/128)`. Per call the running angle
    /// starts at the position and is multiplied by `θ^(−2/64)` after each pair (`ggml_rope_cache_init`),
    /// a pair turns by `freq_scale` times it in f32, and its cos and sin are taken in f64 and
    /// rounded to f32, as the card's table takes them (`RopeTable128::interleaved`).
    fn ggml_rope_head(head: &mut [u16], (y, x): (usize, usize)) {
        let pow = |e: f32| f64::from(ROPE_THETA).powf(f64::from(e)) as f32;
        let (theta_scale, odd) = (pow(-2.0 / 64.0), pow(-2.0 / 128.0));
        for (half, pos, freq_scale) in [(0usize, y, 1.0f32), (1, x, odd)] {
            let mut theta = pos as f32;
            for i in 0..HEAD_DIM_128 / 4 {
                let a = freq_scale * theta;
                let at = HEAD_DIM_128 / 2 * half + 2 * i;
                let (c, s) = (f64::from(a).cos() as f32, f64::from(a).sin() as f32);
                let (y0, y1) = rope_pair_ref(head[at], head[at + 1], c, s);
                head[at] = y0;
                head[at + 1] = y1;
                theta *= theta_scale;
            }
        }
    }

    /// The rows the text model reads, from the projector's `nx·ny` token rows and the break row:
    /// grid row `r`'s tokens, then the break row when `r` is not the last.
    fn rows_rule(proj: &[u16], brk: &[u16], (nx, ny): (usize, usize)) -> Vec<u16> {
        let out = brk.len();
        let mut v = Vec::with_capacity(rows_of(nx, ny) * out);
        for r in 0..ny {
            v.extend_from_slice(&proj[r * nx * out..(r + 1) * nx * out]);
            if r + 1 < ny {
                v.extend_from_slice(brk);
            }
        }
        v
    }

    // ------------------------------------------------------------ host weights

    /// The host copies of the weights block 0 and the projector read, bf16 as the card holds them,
    /// matrices in the file's own order.
    struct HostW {
        patch_w: Vec<u16>,
        pre_ln: Vec<f32>,
        ln1: Vec<f32>,
        /// The file's query, key and value matrices, one after the other: `[3·dim, dim]`.
        qkv_w: Vec<u16>,
        o_w: Vec<u16>,
        ln2: Vec<f32>,
        /// The gate's rows then the up's.
        gu_w: Vec<u16>,
        down_w: Vec<u16>,
        input_norm: Vec<f32>,
        /// `[dim, merge²·dim]`, each row's K in `im2col`'s order.
        merger_w: Vec<u16>,
        mm1: Vec<u16>,
        mm2: Vec<u16>,
        brk: Vec<u16>,
    }

    fn host_w(file: &Gguf) -> Result<HostW, GateError> {
        let hf = HostFile(file);
        let (b16, f32s) = (|name: String| hf.bf16(name), |name: String| hf.f32s(name));
        let mut qkv_w = b16(n::attn_q_weight(0))?;
        qkv_w.extend(b16(n::attn_k_weight(0))?);
        qkv_w.extend(b16(n::attn_v_weight(0))?);
        let mut gu_w = b16(n::ffn_gate_weight(0))?;
        gu_w.extend(b16(n::ffn_up_weight(0))?);
        Ok(HostW {
            patch_w: b16(n::patch_embd_weight())?,
            pre_ln: f32s(n::pre_ln_weight())?,
            ln1: f32s(n::ln1_weight(0))?,
            qkv_w,
            o_w: b16(n::attn_out_weight(0))?,
            ln2: f32s(n::ln2_weight(0))?,
            gu_w,
            down_w: b16(n::ffn_down_weight(0))?,
            input_norm: f32s(n::input_norm_weight())?,
            merger_w: b16(n::patch_merger_weight())?,
            mm1: b16(n::mm1_weight())?,
            mm2: b16(n::mm2_weight())?,
            brk: b16(n::img_break())?,
        })
    }

    // ------------------------------------------------------------ host half

    /// The plan and the patches of the host preprocess against set A: the plan is the set's, and
    /// every value of every patch is the bf16 rounding of the reference's `inp_raw` at the pixel it
    /// came from (the reference's planar `[W, H, 3]` image, our patches in merge order).
    fn preproc_check(
        name: &str,
        set: &ClefvisSet,
        patches: &Patches,
        patch: usize,
    ) -> Result<bool, GateError> {
        let info = set.image(name)?;
        let plan = &patches.plan;
        let (h, w) = (patches.n_vit_h, patches.n_vit_w);
        let shape = (plan.best_w, plan.best_h, plan.n_llm_w, plan.n_llm_h)
            == (info.best_w, info.best_h, info.nx, info.ny)
            && (h, w) == (2 * info.ny, 2 * info.nx);
        let raw = set.inp_raw(name)?;
        let (bw, bh) = (info.best_w, info.best_h);
        if raw.len() != bw * bh * 3 {
            return Err(format!("{name}: inp_raw has {} values for {bw}x{bh}x3", raw.len()).into());
        }
        let plane = patch * patch;
        let mut differ = 0usize;
        for y in 0..h {
            for x in 0..w {
                let m = merge_index(y, x, w);
                let ours = &patches.bf16[m * 3 * plane..(m + 1) * 3 * plane];
                for c in 0..3 {
                    for py in 0..patch {
                        for px in 0..patch {
                            let at = c * bh * bw + (y * patch + py) * bw + x * patch + px;
                            differ +=
                                usize::from(ours[c * plane + py * patch + px] != f32_bf16(raw[at]));
                        }
                    }
                }
            }
        }
        let pass = shape && differ == 0;
        println!(
            "image {name}: preprocess {}x{} (plan {}x{} tokens) equals the set's plan: {}; patches differ from the reference's rounded inp_raw: {differ} of {}  {}",
            plan.best_w,
            plan.best_h,
            plan.n_llm_w,
            plan.n_llm_h,
            verdict(shape),
            patches.bf16.len(),
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------------------ op rules

    /// Every tap the op rules read, besides the last block's and the projector's rows.
    fn rule_taps() -> HashSet<String> {
        let mut w: HashSet<String> = ["embed", "pre_ln", "merger.norm", "merger", "mm1", "gelu"]
            .into_iter()
            .map(str::to_string)
            .collect();
        for op in [
            "norm1", "qkv", "qrot", "krot", "sdpa", "attn", "resid1", "norm2", "gu", "act", "mlp",
        ] {
            w.insert(format!("blk0.{op}"));
        }
        w.insert("blk0".to_string());
        w
    }

    /// Block 0 and the projector of the full-tap image, op by op, on our own tapped inputs.
    #[allow(
        clippy::too_many_lines,
        reason = "one rule per op of the chain, in chain order"
    )]
    fn host_rules(
        hp: &Hparams,
        w: &HostW,
        t: &HashMap<String, Vec<u16>>,
        patches: &[u16],
        grid: (usize, usize),
    ) -> Result<bool, GateError> {
        let (dim, ff, out) = (hp.dim, hp.ff, hp.out_dim);
        let n = grid.0 * grid.1;
        let n_tok = n / (hp.merge * hp.merge);
        let merged = dim * hp.merge * hp.merge;
        let get = |k: &str| t.get(k).ok_or_else(|| format!("tap {k} was not taken"));
        let (embed, pre_ln, norm1, qkv) = (
            get("embed")?,
            get("pre_ln")?,
            get("blk0.norm1")?,
            get("blk0.qkv")?,
        );
        let (qrot, krot, sdpa, attn_side) = (
            get("blk0.qrot")?,
            get("blk0.krot")?,
            get("blk0.sdpa")?,
            get("blk0.attn")?,
        );
        let (resid1, norm2, gu, act) = (
            get("blk0.resid1")?,
            get("blk0.norm2")?,
            get("blk0.gu")?,
            get("blk0.act")?,
        );
        let (mlp_side, blk0, blk_last) = (
            get("blk0.mlp")?,
            get("blk0")?,
            get(&format!("blk{}", hp.n_layer - 1))?,
        );
        let (m_norm, merger, mm1, gelu, proj) = (
            get("merger.norm")?,
            get("merger")?,
            get("mm1")?,
            get("gelu")?,
            get("proj")?,
        );
        let rms_rows = |x: &[u16], g: &[f32]| -> Vec<u16> {
            x.chunks_exact(g.len())
                .flat_map(|r| rms_norm_ref(r, g, hp.eps))
                .collect()
        };
        let residual = |side: &[u16], r: &[u16]| -> Vec<u16> {
            side.iter()
                .zip(r)
                .map(|(&y, &r)| epilogue_ref(y, Epilogue::Residual, r))
                .collect()
        };
        let mut ok = true;

        ok &= gemm_check(
            "patch_embed",
            patches,
            &w.patch_w,
            None,
            n,
            dim,
            3 * hp.patch * hp.patch,
            embed,
        );
        ok &= exact_check("pre_ln", pre_ln, &rms_rows(embed, &w.pre_ln));
        ok &= exact_check("blk0.norm1", norm1, &rms_rows(pre_ln, &w.ln1));

        // The query and key in the file's order of a head: the GEMM against the file's matrices,
        // and the RoPE as llama.cpp turns them.
        let width = 3 * dim;
        let mut qkv_file = qkv.clone();
        for (row, card) in qkv_file
            .chunks_exact_mut(width)
            .zip(qkv.chunks_exact(width))
        {
            row[..dim].copy_from_slice(&heads_to_file(&card[..dim]));
            row[dim..2 * dim].copy_from_slice(&heads_to_file(&card[dim..2 * dim]));
        }
        ok &= gemm_check("blk0.wqkv", norm1, &w.qkv_w, None, n, width, dim, &qkv_file);
        let mut rot = qkv_file.clone();
        for (p, (y, x)) in merge_order(grid.0, grid.1, hp.merge).enumerate() {
            let row = &mut rot[p * width..p * width + 2 * dim];
            for head in row.as_chunks_mut::<HEAD_DIM_128>().0 {
                ggml_rope_head(head, (y, x));
            }
        }
        ok &= exact_check(
            "blk0.rope q",
            &rows_to_file(qrot, dim, 0, dim),
            &cols(&rot, width, 0, dim),
        );
        ok &= exact_check(
            "blk0.rope k",
            &rows_to_file(krot, dim, 0, dim),
            &cols(&rot, width, dim, dim),
        );

        // The attention reads our own rotated q and k (the card's order: the same permutation of
        // both leaves every dot product) and the untouched v.
        let lay = QkvLayout {
            row_width: width,
            q0: 0,
            k0: dim,
            v0: 2 * dim,
            n_heads: hp.n_head,
        };
        let mut ours_rot = qkv.clone();
        for (i, row) in ours_rot.chunks_exact_mut(width).enumerate() {
            row[..dim].copy_from_slice(&qrot[i * dim..(i + 1) * dim]);
            row[dim..2 * dim].copy_from_slice(&krot[i * dim..(i + 1) * dim]);
        }
        let scale = 1.0 / (HEAD_DIM_128 as f64).sqrt();
        let c = par_rows(n, |i| {
            let mut c = Count::default();
            for h in 0..hp.n_head {
                let (o, mag, logit) = attn_128_ref(&ours_rot, lay, n, scale, h, i);
                for d in 0..HEAD_DIM_128 {
                    let y = sdpa[i * dim + h * HEAD_DIM_128 + d];
                    let b = attn_bound(o[d], mag[d], logit, n);
                    c.outside += usize::from(!within_order(y, o[d], b));
                    c.differ += usize::from(y != f32_bf16(o[d] as f32));
                }
            }
            c
        });
        let pass = c.outside == 0;
        println!(
            "rule attn blk0.sdpa     n {n}: outside the f32 bound {} of {}, differ from the exact rounding {} ({:.5})  {}",
            c.outside,
            n * dim,
            c.differ,
            c.differ as f64 / (n * dim) as f64,
            verdict(pass)
        );
        ok &= pass;

        ok &= gemm_check("blk0.wo", sdpa, &w.o_w, None, n, dim, dim, attn_side);
        ok &= exact_check("blk0.resid1 epi", resid1, &residual(attn_side, pre_ln));
        ok &= exact_check("blk0.norm2", norm2, &rms_rows(resid1, &w.ln2));
        ok &= gemm_check("blk0.wgu", norm2, &w.gu_w, None, n, 2 * ff, dim, gu);
        let want: Vec<u16> = gu
            .chunks_exact(2 * ff)
            .flat_map(|r| (0..ff).map(move |j| silu_mul_ref(r[j], r[ff + j])))
            .collect();
        ok &= near_check("blk0.silu_mul", act, &want, SILU_ROWS_PIN);
        ok &= gemm_check("blk0.down", act, &w.down_w, None, n, dim, ff, mlp_side);
        ok &= exact_check("blk0 resid2 epi", blk0, &residual(mlp_side, resid1));

        // The projector: the RMS norm of the last block's rows, the merger as llama.cpp's `im2col`
        // lays a group out (K = c·merge² + ky·merge + kx over the group's merge × merge patches,
        // which are rows merge²·t .. merge²·(t + 1) of the merge order, patch `ky·merge + kx`
        // of them), the GELU MLP, and the rows with their breaks.
        ok &= exact_check("merger.norm", m_norm, &rms_rows(blk_last, &w.input_norm));
        let m2 = hp.merge * hp.merge;
        let mut im2col = vec![0u16; n_tok * merged];
        for (t, group) in m_norm.chunks_exact(m2 * dim).enumerate() {
            for (k, patch) in group.chunks_exact(dim).enumerate() {
                for (c, &v) in patch.iter().enumerate() {
                    im2col[t * merged + c * m2 + k] = v;
                }
            }
        }
        ok &= gemm_check(
            "merger",
            &im2col,
            &w.merger_w,
            None,
            n_tok,
            dim,
            merged,
            merger,
        );
        ok &= gemm_check("mm1", merger, &w.mm1, None, n_tok, out, dim, mm1);
        let want: Vec<u16> = mm1.iter().map(|&x| gelu_tanh_ref(x)).collect();
        ok &= near_check("mm gelu_tanh", gelu, &want, GELU_ROWS_PIN);
        ok &= gemm_check("mm2", gelu, &w.mm2, None, n_tok, out, out, proj);
        Ok(ok)
    }

    // ------------------------------------------------------------ run

    struct Img {
        patches: Patches,
        /// The patch grid `(n_h, n_w)`.
        grid: (usize, usize),
        /// The merged grid `(nx, ny)`.
        tokens: (usize, usize),
    }

    fn load_image(hp: &Hparams, limits: TokenLimits, name: &str) -> Result<Img, GateError> {
        let rule = hp.size_rule(limits)?;
        let patches = preprocess(&read_image(name)?, &rule, &hp.stats)?;
        let grid = (patches.n_vit_h, patches.n_vit_w);
        let tokens = (patches.plan.n_llm_w, patches.plan.n_llm_h);
        Ok(Img {
            patches,
            grid,
            tokens,
        })
    }

    #[allow(
        clippy::too_many_lines,
        reason = "the gate's clauses in chain order, one image at a time"
    )]
    pub fn run() -> Result<(), GateError> {
        let set = ClefvisSet::open(&TAPS.path(TAPS_SET), &TAPS)?;
        // The op rules and the reuse-after-largest clause run on `FULL`: a set without it is refused.
        set.image(FULL)?;
        let pre = ClefvisSet::open(&PREPROC.path(PREPROC_SET), &PREPROC)?;
        let file = Gguf::open(MMPROJ)?;
        let (ctx, stream) = bloomery_gpu::capsync::fresh_stream(0)?;
        let hp0 = Hparams::read(&file)?;
        let limits = hp0.default_limits()?;
        let mut enc = Encoder::load(&ctx, &stream, &file, limits)?;
        let hp = enc.hparams().clone();
        let (dim, out_dim) = (hp.dim, hp.out_dim);
        let last = format!("blk{}", hp.n_layer - 1);
        let mut ok = true;
        println!(
            "load: {} blocks, dim {}, heads {}, ff {}, out {}; weights on the card {} B; activations {} B at {} tokens; {} launches per image of two grid rows or more",
            hp.n_layer,
            dim,
            hp.n_head,
            hp.ff,
            out_dim,
            enc.weight_bytes(),
            enc.scratch_bytes(),
            limits.max(),
            Encoder::launches(hp.n_layer, 2)
        );
        let bytes_ok = enc.weight_bytes() == WEIGHT_BYTES
            && enc.weight_bytes() == weight_bytes_of(&hp)
            && enc.scratch_bytes() == SCRATCH_BYTES
            && enc.scratch_bytes() == scratch_bytes_of(&hp, limits)
            && limits.max() == 1024
            && Encoder::launches(hp.n_layer, 2) == LAUNCHES
            && Encoder::launches(hp.n_layer, 1) == LAUNCHES - 1;
        println!(
            "bytes: weights {} (want {WEIGHT_BYTES}), scratch {} (want {SCRATCH_BYTES}), launches {} (want {LAUNCHES}; one less for one grid row): {}",
            enc.weight_bytes(),
            enc.scratch_bytes(),
            Encoder::launches(hp.n_layer, 2),
            verdict(bytes_ok)
        );
        ok &= bytes_ok;
        let hw = host_w(&file)?;

        let mut names: Vec<(String, bool)> = set
            .images
            .iter()
            .map(|i| (i.name.clone(), i.name == FULL))
            .collect();
        names.sort_by_key(|(n, full)| (!*full, n.clone()));
        let mut full_state: Option<(Img, Vec<u16>)> = None;

        for (name, full) in &names {
            let img = load_image(&hp, limits, name)?;
            let info = set.image(name)?;
            let (nx, ny) = img.tokens;
            let shape = (info.ny * 2, info.nx * 2) == img.grid
                && (info.nx, info.ny) == (nx, ny)
                && info.n_tokens == rows_of(nx, ny);
            println!(
                "image {name}: {}x{} patches, {nx}x{ny} tokens, {} rows; the plan and the rows equal the set's: {}",
                img.grid.0,
                img.grid.1,
                rows_of(nx, ny),
                verdict(shape)
            );
            ok &= shape;
            ok &= preproc_check(name, &pre, &img.patches, hp.patch)?;
            let mut want = HashSet::from([last.clone(), "proj".to_string()]);
            if *full {
                want.extend(rule_taps());
            }
            let mut sink = Collect::new(want);
            let (out, tapped_launches) = {
                let r = enc.encode_tapped(&img.patches, &mut sink)?;
                stream.synchronize()?;
                (r.rows.buf().to_host_vec(&stream)?, r.launches)
            };

            // The sensitivity run: the same encode with one patch value in sixteen nudged.
            let mut noisy = img.patches.clone();
            nudge(&mut noisy.bf16);
            let mut nsink = sink.again();
            let nout = {
                let r = enc.encode_tapped(&noisy, &mut nsink)?;
                stream.synchronize()?;
                r.rows.buf().to_host_vec(&stream)?
            };

            let vit = sink
                .got
                .get(&last)
                .ok_or_else(|| format!("tap {last} was not taken"))?;
            let vit_ref = to_merge(&set.tap(name, PIXTRAL_VIT_OUT)?.1, dim, img.grid)?;
            ok &= band_check(
                K_TAP,
                name,
                PIXTRAL_VIT_OUT,
                vit,
                &vit_ref,
                0.0,
                floor(&vit_ref),
                Some(sensitivity(&nsink.got[&last], vit)),
            );
            let (_, embd_ref) = set.tap(name, EMBD)?;
            ok &= band_check(
                K_TAP,
                name,
                "embd",
                &out,
                &embd_ref,
                0.0,
                floor(&embd_ref),
                Some(sensitivity(&nout, &out)),
            );

            // The projector from the reference's ViT output. Its control is the larger of the
            // rounding floor of the rows and the step's own sensitivity: the same step on its
            // input with one value in sixteen nudged by an ulp.
            let x = to_bf16(&vit_ref);
            let (_, rows) = enc.forced_tail(img.grid, &x)?;
            let (_, nrows) = enc.forced_tail(img.grid, &nudged(&x))?;
            ok &= band_check(
                K_TAP,
                name,
                "forced.tail embd",
                &rows,
                &embd_ref,
                0.0,
                floor(&embd_ref),
                Some(dist(&nrows, &to_f32(&rows)).rms_rel),
            );

            let proj = sink
                .got
                .get("proj")
                .ok_or_else(|| "tap proj was not taken".to_string())?;
            ok &= exact_check(
                &format!("{name} rows"),
                &out,
                &rows_rule(proj, &hw.brk, (nx, ny)),
            );

            if *full {
                ok &= host_rules(&hp, &hw, &sink.got, &img.patches.bf16, img.grid)?;
            }

            let va = {
                let a = enc.encode(&img.patches)?;
                (a.rows.buf().to_host_vec(&stream)?, a.launches)
            };
            let vb = enc.encode(&img.patches)?.rows.buf().to_host_vec(&stream)?;
            ok &= eager_twice(
                name,
                (&out, tapped_launches),
                &va,
                &vb,
                Encoder::launches(hp.n_layer, ny),
            );
            if *full {
                full_state = Some((img, out));
            }
        }

        // An encode does not depend on the larger image encoded before it in the reused buffers.
        if let Some((small, first)) = full_state {
            let largest = load_image(&hp, limits, LARGEST)?;
            let big = enc
                .encode(&largest.patches)?
                .rows
                .buf()
                .to_host_vec(&stream)?;
            let again = enc
                .encode(&small.patches)?
                .rows
                .buf()
                .to_host_vec(&stream)?;
            // The rows of the largest image: its tokens and its break rows.
            let rows = rows_of(largest.tokens.0, largest.tokens.1);
            ok &= reuse_after_largest(
                (FULL, LARGEST),
                (rows, limits.max(), out_dim),
                (&first, &big, &again),
            );
        }

        if ok {
            println!("{NAME}: PASS");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}

#[cfg(not(feature = "vision"))]
fn main() {}
