//! GPU gate for the RADIO image tower on the card (`bloomery_gpu_vision::radio`, the `nemotron_omni`
//! projector of Nemotron 3.5 Bolt): the whole chain from an image's patches to its projected rows, tap by
//! tap against llama.cpp mainline's mtmd graph (the sets of `just dump-ref-boltvis`: set A the
//! preprocess, set B the nodes), and every kernel of block 0 and of the projector against its host rule.
//!
//! The oracle is a CPU build, so a set has no card run and no `.cpu` twin: a tap's controls are the
//! reference rounded to bf16 against itself (the floor of any bf16 tower) and the network's
//! sensitivity — our own encode against our encode with the last bit of one patch value in sixteen
//! flipped. The reference's rows are the 10 class and register rows and then the patches in raster
//! order; ours are the patches in merge order and then the 10 rows, so a tap is mapped to our order
//! before it is compared ([`to_ours`]).
//!
//! Per image of set B, in chain order (the first failing line names the earliest op):
//! 1. The host half: the plan and the patches of `vision::arch::radio::preprocess` against set A's
//!    `# image` line and `inp_raw`, bit for bit after the bf16 rounding.
//! 2. Every tap the set holds (`inp_pos`, blocks 0, 1, 15 and 31 of `layer_inp_normed`, `Qcur`,
//!    `attn_out`, `ffn_inp`, `ffn_out` and `layer_out`, `embd`) against the reference's f32, as
//!    `rms(ours − ref) / rms(ref)`, within [`K_TAP`] times the larger of its controls.
//! 3. Teacher-forced steps: block 0 from `inp_pos` and the class rows, block 1 from `layer_out-0`,
//!    the projector from `layer_out-31`; the band is [`K_TAP`] times the larger of the rounding floor
//!    and the step's own sensitivity.
//! 4. On the set's smallest in-budget image that is no resize (the full-tap `FULL` image), block 0 and
//!    the projector op by op against the host rules of `bloomery_gpu_vision`, each on our own tapped
//!    inputs: the position rows, the norms and the epilogues bit for bit; every GEMM inside its
//!    accumulation-order bound; the attention (heads of 80, the ten class rows among the keys) inside
//!    its f32 bound; the erf GELU inside the libm bound of the glm5v tower's derivation; the ReLU²
//!    bit for bit.
//! 5. Launches per image, two plain encodes bit-identical to each other and to the tapped run.
//!
//! The largest image of the set (embeddings only, over the tap limit) is held to its embeddings.
//!
//! Then once: the ReLU² of every bf16 value against its host rule (NaN stays NaN); the position rows of
//! the table's own grid are the table, and non-square grids match the rule; an encode does not depend
//! on the larger image encoded before it in the reused buffers; the bytes on the card equal the card
//! figure.

#[cfg(feature = "vision")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_radio_tower", gate::run())
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

    use super::vision_rules::{Count, erf_check, exact_check, gemm_check, norm_rows, par_rows};
    use super::vision_taps::{
        Collect, HostFile, band_check, dist, eager_twice, floor, nudge, nudged, read_image,
        reuse_after_largest, sens as sensitivity, to_bf16, to_f32,
    };
    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use bloomery_gpu_vision::attn::{QkvLayout, attn_bound};
    use bloomery_gpu_vision::attn_h80::{HEAD_DIM_80, attn_80_ref};
    use bloomery_gpu_vision::gemm_bf16::{Epilogue, epilogue_ref, within_order};
    use bloomery_gpu_vision::norm::rms_norm_ref;
    use bloomery_gpu_vision::pos_half::{PosHalfKernels, pos_half_ref};
    use bloomery_gpu_vision::radio::{Encoder, scratch_bytes_of, weight_bytes_of};
    use bloomery_gpu_vision::relu_sq::{ReluSqKernels, relu_sq_ref};
    use bloomery_gpu_vision::{bf16_f32, f32_bf16};
    use cuda_core::DeviceBuffer;
    use gguf::Gguf;
    use refset::arch::nemotron_h_moe::vis::{MMPROJ, PREPROC, TAPS};
    use refset::clefvis::{ClefvisSet, EMBD, RADIO_POS, RADIO_TAP_BLOCKS as TAP_BLOCKS, TapScope};
    use vision::Patches;
    use vision::arch::radio::{
        Hparams, POS_SIDE, PREFIX_ROWS, PROJECTOR_HIDDEN, TokenLimits, names as n, preprocess,
    };

    const NAME: &str = "gate_radio_tower";
    const PREPROC_SET: &str = "ref_nemotron_h_moe_vis_preproc";
    const TAPS_SET: &str = "ref_nemotron_h_moe_vis_taps";
    /// The set's image whose block 0 and projector the op rules run on: in the budget and a
    /// multiple of 32, so no resize touches it.
    const FULL: &str = "free-640x480";
    /// The set's image with the largest grid the token limit allows (`embd` only in the set).
    const LARGEST: &str = "max-2600x1800";
    /// The card figures of the file's tower, from its tensor table (`card::weights`: the file's
    /// 1,733,662,720 B of tensor data less the video kernel, the class rows and the patch kernel
    /// narrowed to bf16) and the scratch at the file's 3,328-token limit (178,274 B per token and
    /// 1,540 B more for the ten prefix rows of every stream).
    const WEIGHT_BYTES: u64 = 1_723_781_120;
    const SCRATCH_BYTES: u64 = 593_251_328;
    /// A tap passes when `rms(ours − ref) / rms(ref)` is at most this multiple of the larger of its
    /// controls.
    /// PIN(2026-10-18): fixed before the first run of this tower against its set: the Qwen3-VL
    /// tower's multiple, for the same chain of bf16 ops (32 blocks instead of 27); the design's
    /// prediction is a block output at about 1.6× the rounding floor, and 4 leaves room for a
    /// free-running network's amplification without admitting an order-one defect. First
    /// measurement, same day, nine images: the 200 free-running taps 0.43 to 3.96 (the highest the
    /// 1000x1 strip's `blk0.mlp`, whose control is the rounding floor alone), the 24 forced taps
    /// 1.30 to 2.19, the nine `embd` rows 0.50 to 1.14; the multiple is unchanged.
    const K_TAP: f64 = 4.0;
    /// The RMS norm epsilon of the projector, a constant of llama.cpp's graph (`build_norm(…, 1e-5)`).
    const MLP_NORM_EPS: f32 = 1e-5;

    // ------------------------------------------------------------ taps

    /// One oracle node and the tap of ours that holds it.
    struct Oracle {
        node: String,
        tap: String,
        /// Whether the node's rows start with the ten class and register rows.
        prefix: bool,
        /// The width of one row of our tap and the columns of it the node holds.
        width: usize,
        cols: std::ops::Range<usize>,
    }

    /// Every node the set holds for blocks [`TAP_BLOCKS`], in chain order; the final rows (`embd`)
    /// are the encode's own output.
    fn oracle_taps(dim: usize) -> Vec<Oracle> {
        let mut t = vec![Oracle {
            node: RADIO_POS.to_string(),
            tap: "embed".to_string(),
            prefix: false,
            width: dim,
            cols: 0..dim,
        }];
        for b in TAP_BLOCKS {
            for (node, op, width) in [
                ("layer_inp_normed", "norm1", dim),
                // the query: the first `dim` columns of the fused q, k, v row
                ("Qcur", "qkv", 3 * dim),
                ("attn_out", "attn", dim),
                ("ffn_inp", "resid1", dim),
                ("ffn_out", "mlp", dim),
            ] {
                t.push(Oracle {
                    node: format!("{node}-{b}"),
                    tap: format!("blk{b}.{op}"),
                    prefix: true,
                    width,
                    cols: 0..dim,
                });
            }
            t.push(Oracle {
                node: format!("layer_out-{b}"),
                tap: format!("blk{b}"),
                prefix: true,
                width: dim,
                cols: 0..dim,
            });
        }
        t
    }

    /// Everything our encode is asked to tap for an image: the oracle's taps, and for the full-tap
    /// image the block-0 and projector ops the host rules read.
    fn wanted(dim: usize, full: bool) -> HashSet<String> {
        let mut w: HashSet<String> = oracle_taps(dim).into_iter().map(|o| o.tap).collect();
        if full {
            for op in ["sdpa", "norm2", "up", "act"] {
                w.insert(format!("blk0.{op}"));
            }
            for t in [
                "pos",
                "embed.patch",
                "vit",
                "merger.norm",
                "merger.pre",
                "merger.h",
            ] {
                w.insert(t.to_string());
            }
        }
        w
    }

    // ------------------------------------------------------------ row order

    /// The index, in merge order, of the patch at row `y`, column `x` of a grid `w` patches wide:
    /// the groups of 2×2 patches in raster order, the group's patches in raster order inside it.
    fn merge_index(y: usize, x: usize, w: usize) -> usize {
        ((y / 2) * (w / 2) + x / 2) * 4 + (y % 2) * 2 + x % 2
    }

    /// The oracle's rows of width `dim` in our order: its patches (raster order) in merge order
    /// and, when it has them, the ten class and register rows it starts with after the patches.
    fn to_ours(
        refv: &[f32],
        dim: usize,
        (h, w): (usize, usize),
        prefix: bool,
    ) -> Result<Vec<f32>, GateError> {
        let skip = if prefix { PREFIX_ROWS } else { 0 };
        let patches = h * w;
        if refv.len() != (skip + patches) * dim {
            return Err(format!(
                "{} values for {h}x{w} patches of {dim} and {skip} prefix rows",
                refv.len()
            )
            .into());
        }
        let mut out = vec![0.0f32; refv.len()];
        for y in 0..h {
            for x in 0..w {
                let src = (skip + y * w + x) * dim;
                let dst = merge_index(y, x, w) * dim;
                out[dst..dst + dim].copy_from_slice(&refv[src..src + dim]);
            }
        }
        out[patches * dim..].copy_from_slice(&refv[..skip * dim]);
        Ok(out)
    }

    /// The columns `cols` of every row of `width` of `bits`.
    fn columns(bits: &[u16], width: usize, cols: &std::ops::Range<usize>) -> Vec<u16> {
        if cols.len() == width {
            return bits.to_vec();
        }
        bits.chunks_exact(width)
            .flat_map(|r| r[cols.clone()].iter().copied())
            .collect()
    }

    // ------------------------------------------------------------ host weights

    /// The host copies of the weights block 0 and the projector read, bf16 as the card holds them.
    struct HostW {
        pos: Vec<f32>,
        patch_w: Vec<u16>,
        prefix: Vec<u16>,
        ln1: (Vec<f32>, Vec<f32>),
        qkv_w: Vec<u16>,
        qkv_b: Vec<f32>,
        o_w: Vec<u16>,
        o_b: Vec<f32>,
        ln2: (Vec<f32>, Vec<f32>),
        up_w: Vec<u16>,
        up_b: Vec<f32>,
        down_w: Vec<u16>,
        down_b: Vec<f32>,
        mlp_norm: Vec<f32>,
        mm1: Vec<u16>,
        mm3: Vec<u16>,
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
            pos: f32s(n::position_embd())?,
            patch_w: b16(n::patch_embd_weight())?,
            prefix: b16(n::class_embd())?,
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
            mlp_norm: f32s(n::mlp_norm())?,
            mm1: b16(n::mlp_up())?,
            mm3: b16(n::mlp_down())?,
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
        out: &[u16],
    ) -> Result<bool, GateError> {
        let (dim, ff, n_patch) = (hp.dim, hp.ff, grid.0 * grid.1);
        let rows = n_patch + PREFIX_ROWS;
        let n_tok = n_patch / (hp.merge * hp.merge);
        let merged = hp.dim * hp.merge * hp.merge;
        let get = |k: &str| t.get(k).ok_or_else(|| format!("tap {k} was not taken"));
        let (pos, embed_patch, embed) = (get("pos")?, get("embed.patch")?, get("embed")?);
        let (norm1, qkv, sdpa, attn_side) = (
            get("blk0.norm1")?,
            get("blk0.qkv")?,
            get("blk0.sdpa")?,
            get("blk0.attn")?,
        );
        let (resid1, norm2, up, act) = (
            get("blk0.resid1")?,
            get("blk0.norm2")?,
            get("blk0.up")?,
            get("blk0.act")?,
        );
        let (mlp_side, blk0, blk_last) = (
            get("blk0.mlp")?,
            get("blk0")?,
            get(&format!("blk{}", hp.n_layer - 1))?,
        );
        let (vit, m_norm, m_pre, m_h) = (
            get("vit")?,
            get("merger.norm")?,
            get("merger.pre")?,
            get("merger.h")?,
        );
        let k_patch = 3 * hp.patch * hp.patch;
        let mut ok = true;

        ok &= exact_check("pos rows", pos, &pos_half_ref(&w.pos, POS_SIDE, dim, grid));
        ok &= gemm_check(
            "patch_embed",
            patches,
            &w.patch_w,
            None,
            n_patch,
            dim,
            k_patch,
            embed_patch,
        );
        let want: Vec<u16> = embed_patch
            .iter()
            .zip(pos)
            .map(|(&y, &r)| epilogue_ref(y, Epilogue::Residual, r))
            .collect();
        ok &= exact_check("embed epi", embed, &want);
        // The stream of block 0: the patches' rows and the class rows behind them.
        let mut stream0 = embed.clone();
        stream0.extend(&w.prefix);
        ok &= exact_check("blk0.norm1", norm1, &norm_rows(&stream0, &w.ln1, hp.eps));
        ok &= gemm_check(
            "blk0.wqkv",
            norm1,
            &w.qkv_w,
            Some(&w.qkv_b),
            rows,
            3 * dim,
            dim,
            qkv,
        );
        let lay = QkvLayout {
            row_width: 3 * dim,
            q0: 0,
            k0: dim,
            v0: 2 * dim,
            n_heads: hp.n_head,
        };
        let scale = 1.0 / (HEAD_DIM_80 as f64).sqrt();
        let c = par_rows(rows, |i| {
            let mut c = Count::default();
            for h in 0..hp.n_head {
                let (o, mag, logit) = attn_80_ref(qkv, lay, rows, scale, h, i);
                for d in 0..HEAD_DIM_80 {
                    let y = sdpa[i * dim + h * HEAD_DIM_80 + d];
                    let b = attn_bound(o[d], mag[d], logit, rows);
                    c.outside += usize::from(!within_order(y, o[d], b));
                    c.differ += usize::from(y != f32_bf16(o[d] as f32));
                }
            }
            c
        });
        let pass = c.outside == 0;
        println!(
            "rule attn blk0.sdpa     n {rows}: outside the f32 bound {} of {}, differ from the exact rounding {} ({:.5})  {}",
            c.outside,
            rows * dim,
            c.differ,
            c.differ as f64 / (rows * dim) as f64,
            verdict(pass)
        );
        ok &= pass;

        ok &= gemm_check(
            "blk0.wo",
            sdpa,
            &w.o_w,
            Some(&w.o_b),
            rows,
            dim,
            dim,
            attn_side,
        );
        let want: Vec<u16> = attn_side
            .iter()
            .zip(&stream0)
            .map(|(&y, &r)| epilogue_ref(y, Epilogue::Residual, r))
            .collect();
        ok &= exact_check("blk0.resid1 epi", resid1, &want);
        ok &= exact_check("blk0.norm2", norm2, &norm_rows(resid1, &w.ln2, hp.eps));
        ok &= gemm_check("blk0.up", norm2, &w.up_w, Some(&w.up_b), rows, ff, dim, up);
        ok &= erf_check("blk0.gelu_erf", up, act);
        ok &= gemm_check(
            "blk0.down",
            act,
            &w.down_w,
            Some(&w.down_b),
            rows,
            dim,
            ff,
            mlp_side,
        );
        let want: Vec<u16> = mlp_side
            .iter()
            .zip(resid1)
            .map(|(&y, &r)| epilogue_ref(y, Epilogue::Residual, r))
            .collect();
        ok &= exact_check("blk0 resid2 epi", blk0, &want);

        // The projector: the class rows are dropped (there is no final norm), the pixel shuffle is
        // a re-read of the merge-order rows as rows of 4·dim, the norm is an RMS norm of them.
        ok &= exact_check(
            "vit = last block's patches",
            vit,
            &blk_last[..n_patch * dim],
        );
        let want: Vec<u16> = vit
            .chunks_exact(merged)
            .flat_map(|r| rms_norm_ref(r, &w.mlp_norm, MLP_NORM_EPS))
            .collect();
        ok &= exact_check("merger.norm", m_norm, &want);
        ok &= gemm_check(
            "merger.mm1",
            m_norm,
            &w.mm1,
            None,
            n_tok,
            PROJECTOR_HIDDEN,
            merged,
            m_pre,
        );
        let want: Vec<u16> = m_pre.iter().map(|&y| relu_sq_ref(y)).collect();
        ok &= exact_check("merger relu_sq", m_h, &want);
        ok &= gemm_check(
            "merger.mm3",
            m_h,
            &w.mm3,
            None,
            n_tok,
            hp.out_dim,
            PROJECTOR_HIDDEN,
            out,
        );
        Ok(ok)
    }

    // ------------------------------------------------------------ once-per-run rules

    /// ReLU² on every bf16 value against the host rule: bit for bit, a NaN staying a NaN.
    fn relu_sweep(
        ctx: &std::sync::Arc<cuda_core::CudaContext>,
        stream: &cuda_core::CudaStream,
    ) -> Result<bool, GateError> {
        let xs: Vec<u16> = (0..=u16::MAX).collect();
        let k = ReluSqKernels::load(ctx)?;
        let mut d = DeviceBuffer::from_host(stream, &xs)?;
        k.enqueue(stream, xs.len(), &mut d)?;
        stream.synchronize()?;
        let got = d.to_host_vec(stream)?;
        let differ = xs
            .iter()
            .zip(&got)
            .filter(|&(&x, &g)| {
                let want = relu_sq_ref(x);
                if bf16_f32(want).is_nan() {
                    !bf16_f32(g).is_nan()
                } else {
                    g != want
                }
            })
            .count();
        let pass = differ == 0;
        println!(
            "rule relu_sq sweep      every bf16 pattern: {differ} of {} differ from the host rule (a NaN stays a NaN)  {}",
            xs.len(),
            verdict(pass)
        );
        Ok(pass)
    }

    /// The position rows of the table's own grid are the table in merge order (llama.cpp returns
    /// the table unresized when the grid is the table's, and a half-pixel resize at scale 1 is the
    /// identity), and the grids of a strip, a small image and a non-square one match the rule.
    fn pos_identity(
        ctx: &std::sync::Arc<cuda_core::CudaContext>,
        stream: &cuda_core::CudaStream,
        w: &HostW,
        dim: usize,
    ) -> Result<bool, GateError> {
        let k = PosHalfKernels::load(ctx)?;
        let table = DeviceBuffer::from_host(stream, &w.pos)?;
        let mut ok = true;
        for grid in [(POS_SIDE, POS_SIDE), (6usize, 10usize), (2, 1012)] {
            let mut out = DeviceBuffer::<u16>::zeroed(stream, grid.0 * grid.1 * dim)?;
            k.enqueue(stream, &table, (POS_SIDE, dim), grid, &mut out)?;
            stream.synchronize()?;
            let got = out.to_host_vec(stream)?;
            ok &= exact_check(
                &format!("pos {}x{} host", grid.0, grid.1),
                &got,
                &pos_half_ref(&w.pos, POS_SIDE, dim, grid),
            );
            if grid == (POS_SIDE, POS_SIDE) {
                let mut want = Vec::new();
                for gy in 0..POS_SIDE / 2 {
                    for gx in 0..POS_SIDE / 2 {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let p = (2 * gy + dy) * POS_SIDE + 2 * gx + dx;
                                want.extend(
                                    w.pos[p * dim..(p + 1) * dim].iter().map(|&v| f32_bf16(v)),
                                );
                            }
                        }
                    }
                }
                ok &= exact_check("pos 128x128 = table", &got, &want);
            }
        }
        Ok(ok)
    }

    // ------------------------------------------------------------ run

    struct Img {
        patches: Patches,
        grid: (usize, usize),
    }

    fn load_image(hp: &Hparams, limits: TokenLimits, name: &str) -> Result<Img, GateError> {
        let rule = hp.size_rule(limits)?;
        let patches = preprocess(&read_image(name)?, &rule, &hp.stats)?;
        let grid = (patches.n_vit_h, patches.n_vit_w);
        Ok(Img { patches, grid })
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
        let want_launches = Encoder::launches(hp.n_layer);
        let dim = hp.dim;
        let mut ok = true;
        println!(
            "load: {} blocks, dim {}, heads {}, ff {}, out {}; weights on the card {} B; activations {} B at {} tokens; {} launches per image",
            hp.n_layer,
            dim,
            hp.n_head,
            hp.ff,
            hp.out_dim,
            enc.weight_bytes(),
            enc.scratch_bytes(),
            limits.max(),
            want_launches
        );
        let bytes_ok = enc.weight_bytes() == WEIGHT_BYTES
            && enc.weight_bytes() == weight_bytes_of(&hp)
            && enc.scratch_bytes() == SCRATCH_BYTES
            && enc.scratch_bytes() == scratch_bytes_of(&hp, limits)
            && limits.max() == 3328
            && want_launches == 230;
        println!(
            "bytes: weights {} (want {WEIGHT_BYTES}), scratch {} (want {SCRATCH_BYTES}), launches {want_launches} (want 230): {}",
            enc.weight_bytes(),
            enc.scratch_bytes(),
            verdict(bytes_ok)
        );
        ok &= bytes_ok;
        let hw = host_w(&file)?;
        ok &= relu_sweep(&ctx, &stream)?;
        ok &= pos_identity(&ctx, &stream, &hw, dim)?;

        let mut names: Vec<(String, bool)> = set
            .images
            .iter()
            .map(|i| (i.name.clone(), i.name == FULL))
            .collect();
        names.sort_by_key(|(n, full)| (!*full, n.clone()));
        let oracle = oracle_taps(dim);
        let mut full_state: Option<(Img, Vec<u16>)> = None;

        for (name, full) in &names {
            let img = load_image(&hp, limits, name)?;
            let info = set.image(name)?;
            let shape = (info.ny * 2, info.nx * 2) == img.grid;
            println!(
                "image {name}: {}x{} patches, {} tokens; the plan equals the set's: {}",
                img.grid.0,
                img.grid.1,
                img.grid.0 * img.grid.1 / 4,
                verdict(shape)
            );
            ok &= shape;
            ok &= preproc_check(name, &pre, &img.patches, hp.patch)?;
            let taps_full = set.tap_effect_of(name)?.scope == TapScope::Full;
            let mut sink = Collect::new(if taps_full {
                wanted(dim, *full)
            } else {
                HashSet::new()
            });
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
            let embd_sens = sensitivity(&nout, &out);

            if taps_full {
                for o in &oracle {
                    let bits = sink
                        .got
                        .get(&o.tap)
                        .ok_or_else(|| format!("tap {} was not taken", o.tap))?;
                    let ours = columns(bits, o.width, &o.cols);
                    let (_, refv) = set.tap(name, &o.node)?;
                    let refv = to_ours(&refv, dim, img.grid, o.prefix)?;
                    // Block 0's taps are within one block of the patch embedding, so the
                    // sensitivity control applies from block 1 on.
                    let free = !(o.tap == "embed" || o.tap.starts_with("blk0."));
                    let s = free.then(|| {
                        let noisy = columns(&nsink.got[&o.tap], o.width, &o.cols);
                        dist(&noisy, &to_f32(&ours)).rms_rel
                    });
                    ok &= band_check(K_TAP, name, &o.tap, &ours, &refv, 0.0, floor(&refv), s);
                }
            }
            let (_, refv) = set.tap(name, EMBD)?;
            ok &= band_check(
                K_TAP,
                name,
                "embd",
                &out,
                &refv,
                0.0,
                floor(&refv),
                Some(embd_sens),
            );

            if taps_full {
                // Teacher-forced steps from the reference's own inputs. A step's control is the
                // larger of the rounding floor of its output and the step's own sensitivity: the
                // same step run on its input with one value in sixteen nudged by an ulp.
                let get = |node: &str, prefix: bool| -> Result<Vec<f32>, GateError> {
                    to_ours(&set.tap(name, node)?.1, dim, img.grid, prefix)
                };
                let mut x0 = to_bf16(&get(RADIO_POS, false)?);
                x0.extend(&hw.prefix);
                for (b, x) in [(0usize, x0), (1, to_bf16(&get("layer_out-0", true)?))] {
                    let got = enc.forced_block(img.grid, b, &x)?;
                    let nsens =
                        dist(&enc.forced_block(img.grid, b, &nudged(&x))?, &to_f32(&got)).rms_rel;
                    let want = get(&format!("layer_out-{b}"), true)?;
                    ok &= band_check(
                        K_TAP,
                        name,
                        &format!("forced.blk{b}"),
                        &got,
                        &want,
                        0.0,
                        floor(&want),
                        Some(nsens),
                    );
                }
                let last = format!("layer_out-{}", hp.n_layer - 1);
                let x = to_bf16(&get(&last, true)?);
                let (_, rows) = enc.forced_tail(img.grid, &x)?;
                let (_, nrows) = enc.forced_tail(img.grid, &nudged(&x))?;
                let want = set.tap(name, EMBD)?.1;
                ok &= band_check(
                    K_TAP,
                    name,
                    "forced.tail embd",
                    &rows,
                    &want,
                    0.0,
                    floor(&want),
                    Some(dist(&nrows, &to_f32(&rows)).rms_rel),
                );
            }

            if *full {
                ok &= host_rules(&hp, &hw, &sink.got, &img.patches.bf16, img.grid, &out)?;
            }

            let va = {
                let a = enc.encode(&img.patches)?;
                (a.rows.buf().to_host_vec(&stream)?, a.launches)
            };
            let vb = enc.encode(&img.patches)?.rows.buf().to_host_vec(&stream)?;
            ok &= eager_twice(name, (&out, tapped_launches), &va, &vb, want_launches);
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
            let tokens = largest.grid.0 * largest.grid.1 / 4;
            ok &= reuse_after_largest(
                (FULL, LARGEST),
                (tokens, limits.max(), hp.out_dim),
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
