//! GPU gate for Hy4-flash's image tower on the card (`bloomery_gpu_vision::hy4v`, the `hy_v4_flash`
//! projector): the whole chain from an image's patches to the rows the text model reads, against
//! llama.cpp pull request #32517's mtmd graph (the sets of `just dump-ref-hy4vis`: set A the
//! preprocess, set B the last block's `layer_out` and `embd`), and every kernel of block 0 and of the
//! adapter against its host rule.
//!
//! The oracle is a CPU build, so a set has no card run and no `.cpu` twin: a tap's controls are the
//! reference rounded to bf16 against itself (the floor of any bf16 tower) and the network's
//! sensitivity — our own encode against our encode with the last bit of one patch value in sixteen
//! flipped. The reference's ViT rows are the patches in raster order and ours in merge order, so the
//! ViT's output is mapped to our order before it is compared ([`to_merge`]); the rows the text model
//! reads (tokens and newline rows) are in the same order on both sides.
//!
//! Per image of set B, in chain order (the first failing line names the earliest op):
//! 1. The host half: the plan and the patches of `vision::arch::hy4v::preprocess` (the Lanczos
//!    resize) against set A's `# image` line and `inp_raw`, bit for bit after the bf16 rounding, and
//!    the image's rows (`(nx + 1)·ny`) against the set's token count.
//! 2. The ViT's output (`layer_out` of the last block) and the rows (`embd`) against the reference's
//!    f32, as `rms(ours − ref) / rms(ref)`, within [`K_TAP`] times the larger of its controls.
//! 3. The adapter teacher-forced from the reference's ViT output; the band is [`K_TAP`] times the
//!    larger of the rounding floor and the step's own sensitivity.
//! 4. The rows: each grid row's tokens are the adapter's rows and the row after each grid row is the
//!    file's newline row, bit for bit.
//! 5. On the set's image that is no resize (the `FULL` image), the position rows, block 0 and the
//!    adapter op by op against the host rules of `bloomery_gpu_vision`, each on our own tapped
//!    inputs: the position rows against HunyuanVL's rule ([`pos_hunyuan_ref`]) and the norms and the
//!    epilogues bit for bit; every GEMM inside its accumulation-order bound, the query, key and
//!    value against the file's own matrices and the strided convolution against the file's kernel
//!    in `im2col`'s order (the load's reorder not used); the attention (heads of 72) inside its f32
//!    bound; the tanh GELU against its host rule and the erf GELU inside its bound.
//! 6. Launches per image, two plain encodes bit-identical to each other and to the tapped run.
//!
//! Then once: an encode does not depend on the larger image encoded before it in the reused
//! buffers; the bytes on the card equal the card figure.

#[cfg(feature = "vision")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_hy4v_tower", gate::run())
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

    use super::vision_rules::{
        Count, erf_check, exact_check, gemm_check, near_check, norm_rows, par_rows,
    };
    use super::vision_taps::{
        Collect, HostFile, band_check, dist, eager_twice, floor, nudge, nudged, read_image,
        reuse_after_largest, sens as sensitivity, to_bf16, to_f32,
    };
    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use bloomery_gpu_vision::attn::{HEAD_DIM_72, QkvLayout, attn_72_ref, attn_bound};
    use bloomery_gpu_vision::gelu_tanh::gelu_tanh_ref;
    use bloomery_gpu_vision::gemm_bf16::{Epilogue, epilogue_ref, within_order};
    use bloomery_gpu_vision::hy4v::{Encoder, scratch_bytes_of, weight_bytes_of};
    use bloomery_gpu_vision::norm::rms_norm_ref;
    use bloomery_gpu_vision::pos_hunyuan::pos_hunyuan_ref;
    use bloomery_gpu_vision::{bf16_f32, f32_bf16};
    use gguf::Gguf;
    use refset::arch::hy4::vis::{MMPROJ, PREPROC, TAPS};
    use refset::clefvis::{ClefvisSet, EMBD, HY4V_VIT_OUT};
    use vision::Patches;
    use vision::arch::hy4v::card::ff_padded;
    use vision::arch::hy4v::{Hparams, POS_SIDE, TokenLimits, names as n, preprocess, rows_of};

    const NAME: &str = "gate_hy4v_tower";
    const PREPROC_SET: &str = "ref_hy4_vis_preproc";
    const TAPS_SET: &str = "ref_hy4_vis_taps";
    /// The set's image whose position rows, block 0 and adapter the op rules run on: in the budget
    /// and a multiple of 32, so no resize touches it; 14 grid rows, so 14 newline rows.
    const FULL: &str = "free-448x448";
    /// The set's image with the largest grid the token limit allows.
    const LARGEST: &str = "max-2600x1800";
    /// The card figures of the file's tower [derived]: the weights from its tensor table
    /// (`card::weights`: the file's 1,044,371,904 B of tensor data less the patch kernel, the
    /// position table and the two convolutions narrowed from f32 to bf16 and the newline row, plus
    /// the MLP's padding to 4,352), the scratch at the file's 4,096-token limit (`card::ScratchLens`:
    /// 16,384 patches; 148,480 B a token).
    const WEIGHT_BYTES: u64 = 940_039_680;
    const SCRATCH_BYTES: u64 = 608_174_080;
    /// Launches of an image: the position rows and the patch GEMM, eight per block of 27, the
    /// adapter's five.
    const LAUNCHES: usize = 223;
    /// A tap passes when `rms(ours − ref) / rms(ref)` is at most this multiple of the larger of its
    /// controls.
    /// PIN(2026-11-19): fixed before the first run of this tower against its set: the Qwen3-VL,
    /// RADIO and Pixtral towers' multiple, for the same chain of bf16 ops (27 blocks of the Qwen3-VL
    /// tower's shape); 4 leaves room for a free-running network's amplification without admitting
    /// an order-one defect.
    const K_TAP: f64 = 4.0;
    /// Values of block 0's tanh-GELU rows of the full-tap image (784 × 4,304) that may differ from
    /// the host rule, each by one bf16 step at most.
    /// PIN(2026-11-19): fixed before the first run: the same kernel (`vis_gelu_tanh`) measured 0 on
    /// the Qwen3-VL tower's block-0 rows; this gate's first rule line is the evidence for it.
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

    /// The rows the text model reads, from the adapter's `nx·ny` token rows and the newline row:
    /// grid row `r`'s tokens, then the newline row.
    fn rows_rule(proj: &[u16], newline: &[u16], (nx, ny): (usize, usize)) -> Vec<u16> {
        let out = newline.len();
        let mut v = Vec::with_capacity(rows_of(nx, ny) * out);
        for r in 0..ny {
            v.extend_from_slice(&proj[r * nx * out..(r + 1) * nx * out]);
            v.extend_from_slice(newline);
        }
        v
    }

    /// The first `ff` of every row of `width`.
    fn real_cols(bits: &[u16], width: usize, ff: usize) -> Vec<u16> {
        bits.chunks_exact(width)
            .flat_map(|r| r[..ff].iter().copied())
            .collect()
    }

    // ------------------------------------------------------------ host weights

    /// The host copies of the weights the position rows, block 0 and the adapter read, bf16 as the
    /// card holds them, matrices in the file's own order and shape (the MLP unpadded).
    struct HostW {
        patch_w: Vec<u16>,
        patch_b: Vec<f32>,
        pos: Vec<f32>,
        ln1: (Vec<f32>, Vec<f32>),
        /// The file's query, key and value matrices, one after the other: `[3·dim, dim]`.
        qkv_w: Vec<u16>,
        qkv_b: Vec<f32>,
        o_w: Vec<u16>,
        o_b: Vec<f32>,
        ln2: (Vec<f32>, Vec<f32>),
        up_w: Vec<u16>,
        up_b: Vec<f32>,
        down_w: Vec<u16>,
        down_b: Vec<f32>,
        pre_norm: Vec<f32>,
        /// `[4·dim, merge²·dim]`, each row's K in `im2col`'s order.
        conv_w: Vec<u16>,
        conv_b: Vec<f32>,
        proj_w: Vec<u16>,
        proj_b: Vec<f32>,
        newline: Vec<u16>,
    }

    fn host_w(file: &Gguf) -> Result<HostW, GateError> {
        let hf = HostFile(file);
        let (b16, f32s) = (|name: String| hf.bf16(name), |name: String| hf.f32s(name));
        let mut qkv_w = b16(n::attn_q_weight(0))?;
        qkv_w.extend(b16(n::attn_k_weight(0))?);
        qkv_w.extend(b16(n::attn_v_weight(0))?);
        let mut qkv_b = f32s(n::attn_q_bias(0))?;
        qkv_b.extend(f32s(n::attn_k_bias(0))?);
        qkv_b.extend(f32s(n::attn_v_bias(0))?);
        Ok(HostW {
            patch_w: b16(n::patch_embd_weight())?,
            patch_b: f32s(n::patch_embd_bias())?,
            pos: f32s(n::position_embd())?,
            ln1: (f32s(n::ln1_weight(0))?, f32s(n::ln1_bias(0))?),
            qkv_w,
            qkv_b,
            o_w: b16(n::attn_out_weight(0))?,
            o_b: f32s(n::attn_out_bias(0))?,
            ln2: (f32s(n::ln2_weight(0))?, f32s(n::ln2_bias(0))?),
            up_w: b16(n::ffn_up_weight(0))?,
            up_b: f32s(n::ffn_up_bias(0))?,
            down_w: b16(n::ffn_down_weight(0))?,
            down_b: f32s(n::ffn_down_bias(0))?,
            pre_norm: f32s(n::pre_norm_weight())?,
            conv_w: b16(n::conv_weight())?,
            conv_b: f32s(n::conv_bias())?,
            proj_w: b16(n::proj_weight())?,
            proj_b: f32s(n::proj_bias())?,
            newline: b16(n::image_newline())?,
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

    /// Every tap the op rules read, besides the last block's and the adapter's rows.
    fn rule_taps() -> HashSet<String> {
        let mut w: HashSet<String> = [
            "pos",
            "embed.patch",
            "embed",
            "merger.norm",
            "conv.pre",
            "conv",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();
        for op in [
            "norm1", "qkv", "sdpa", "attn", "resid1", "norm2", "up", "act", "mlp",
        ] {
            w.insert(format!("blk0.{op}"));
        }
        w.insert("blk0".to_string());
        w
    }

    /// The position rows, block 0 and the adapter of the full-tap image, op by op, on our own
    /// tapped inputs.
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
        let ff_pad = ff_padded(hp);
        let n = grid.0 * grid.1;
        let n_tok = n / (hp.merge * hp.merge);
        let merged = hp.merged();
        let get = |k: &str| t.get(k).ok_or_else(|| format!("tap {k} was not taken"));
        let (pos, embed_patch, embed, norm1) = (
            get("pos")?,
            get("embed.patch")?,
            get("embed")?,
            get("blk0.norm1")?,
        );
        let (qkv, sdpa, attn_side, resid1) = (
            get("blk0.qkv")?,
            get("blk0.sdpa")?,
            get("blk0.attn")?,
            get("blk0.resid1")?,
        );
        let (norm2, up, act, mlp_side) = (
            get("blk0.norm2")?,
            get("blk0.up")?,
            get("blk0.act")?,
            get("blk0.mlp")?,
        );
        let (blk0, blk_last) = (get("blk0")?, get(&format!("blk{}", hp.n_layer - 1))?);
        let (m_norm, conv_pre, conv, proj) = (
            get("merger.norm")?,
            get("conv.pre")?,
            get("conv")?,
            get("proj")?,
        );
        let residual = |side: &[u16], r: &[u16]| -> Vec<u16> {
            side.iter()
                .zip(r)
                .map(|(&y, &r)| epilogue_ref(y, Epilogue::Residual, r))
                .collect()
        };
        let mut ok = true;

        ok &= exact_check(
            "pos rows",
            pos,
            &pos_hunyuan_ref(&w.pos, POS_SIDE, dim, grid),
        );
        ok &= gemm_check(
            "patch_embed",
            patches,
            &w.patch_w,
            Some(&w.patch_b),
            n,
            dim,
            3 * hp.patch * hp.patch,
            embed_patch,
        );
        ok &= exact_check("embed epi", embed, &residual(embed_patch, pos));
        ok &= exact_check("blk0.norm1", norm1, &norm_rows(embed, &w.ln1, hp.eps));
        let width = 3 * dim;
        ok &= gemm_check(
            "blk0.wqkv",
            norm1,
            &w.qkv_w,
            Some(&w.qkv_b),
            n,
            width,
            dim,
            qkv,
        );

        // The attention reads the query, key and value as the GEMM wrote them: no RoPE.
        let lay = QkvLayout {
            row_width: width,
            q0: 0,
            k0: dim,
            v0: 2 * dim,
            n_heads: hp.n_head,
        };
        let scale = 1.0 / (HEAD_DIM_72 as f64).sqrt();
        let c = par_rows(n, |i| {
            let mut c = Count::default();
            for h in 0..hp.n_head {
                let (o, mag, logit) = attn_72_ref(qkv, lay, n, scale, h, i);
                for d in 0..HEAD_DIM_72 {
                    let y = sdpa[i * dim + h * HEAD_DIM_72 + d];
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

        ok &= gemm_check(
            "blk0.wo",
            sdpa,
            &w.o_w,
            Some(&w.o_b),
            n,
            dim,
            dim,
            attn_side,
        );
        ok &= exact_check("blk0.resid1 epi", resid1, &residual(attn_side, embed));
        ok &= exact_check("blk0.norm2", norm2, &norm_rows(resid1, &w.ln2, hp.eps));
        // The up projection: the file's 4304 columns by the exact product, the padded columns
        // exactly zero.
        let real = real_cols(up, ff_pad, ff);
        let pad_zero = up
            .chunks_exact(ff_pad)
            .all(|r| r[ff..].iter().all(|&v| bf16_f32(v) == 0.0));
        ok &= gemm_check("blk0.up", norm2, &w.up_w, Some(&w.up_b), n, ff, dim, &real);
        println!(
            "rule blk0.up padding    columns {ff}..{ff_pad} are zero: {}",
            verdict(pad_zero)
        );
        ok &= pad_zero;
        let real_act = real_cols(act, ff_pad, ff);
        let want: Vec<u16> = real.iter().map(|&x| gelu_tanh_ref(x)).collect();
        ok &= near_check("blk0.gelu_tanh", &real_act, &want, GELU_ROWS_PIN);
        ok &= gemm_check(
            "blk0.down",
            &real_act,
            &w.down_w,
            Some(&w.down_b),
            n,
            dim,
            ff,
            mlp_side,
        );
        ok &= exact_check("blk0 resid2 epi", blk0, &residual(mlp_side, resid1));

        // The adapter: the RMS norm of the last block's rows, the strided convolution as ggml's
        // `im2col` lays a group out (K = c·merge² + ky·merge + kx over the group's merge × merge
        // patches, which are rows merge²·t .. merge²·(t + 1) of the merge order, patch
        // `ky·merge + kx` of them), its erf GELU, the 1×1 convolution, and the rows with their
        // newlines.
        let rms_rows: Vec<u16> = blk_last
            .chunks_exact(dim)
            .flat_map(|r| rms_norm_ref(r, &w.pre_norm, hp.eps))
            .collect();
        ok &= exact_check("merger.norm", m_norm, &rms_rows);
        let m2 = hp.merge * hp.merge;
        let mut im2col = vec![0u16; n_tok * merged];
        for (tok, group) in m_norm.chunks_exact(m2 * dim).enumerate() {
            for (k, patch) in group.chunks_exact(dim).enumerate() {
                for (c, &v) in patch.iter().enumerate() {
                    im2col[tok * merged + c * m2 + k] = v;
                }
            }
        }
        ok &= gemm_check(
            "conv k2s2",
            &im2col,
            &w.conv_w,
            Some(&w.conv_b),
            n_tok,
            merged,
            merged,
            conv_pre,
        );
        ok &= erf_check("conv gelu_erf", conv_pre, conv);
        ok &= gemm_check(
            "conv 1x1",
            conv,
            &w.proj_w,
            Some(&w.proj_b),
            n_tok,
            out,
            merged,
            proj,
        );
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
            "load: {} blocks, dim {}, heads {}, ff {}, out {}; weights on the card {} B; activations {} B at {} tokens; {} launches per image",
            hp.n_layer,
            dim,
            hp.n_head,
            hp.ff,
            out_dim,
            enc.weight_bytes(),
            enc.scratch_bytes(),
            limits.max(),
            Encoder::launches(hp.n_layer)
        );
        let bytes_ok = enc.weight_bytes() == WEIGHT_BYTES
            && enc.weight_bytes() == weight_bytes_of(&hp)
            && enc.scratch_bytes() == SCRATCH_BYTES
            && enc.scratch_bytes() == scratch_bytes_of(&hp, limits)
            && limits.max() == 4096
            && Encoder::launches(hp.n_layer) == LAUNCHES;
        println!(
            "bytes: weights {} (want {WEIGHT_BYTES}), scratch {} (want {SCRATCH_BYTES}), launches {} (want {LAUNCHES}): {}",
            enc.weight_bytes(),
            enc.scratch_bytes(),
            Encoder::launches(hp.n_layer),
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
            let vit_ref = to_merge(&set.tap(name, HY4V_VIT_OUT)?.1, dim, img.grid)?;
            ok &= band_check(
                K_TAP,
                name,
                HY4V_VIT_OUT,
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

            // The adapter from the reference's ViT output. Its control is the larger of the
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
                &rows_rule(proj, &hw.newline, (nx, ny)),
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
                Encoder::launches(hp.n_layer),
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
            // The rows of the largest image: its tokens and its newline rows.
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
