//! GPU gate for the Qwen3-VL image tower on the card (`bloomery_gpu_vision::qwen3vl`, the
//! `qwen3vl_merger` projector of Clef-Flash, Qwen3.6-35B-A3B and Qwen3.8-Flash-Next): the whole
//! chain from an image's patches to its merger rows, tap by tap against llama.cpp mainline's mtmd
//! graph (the clefvis set B, `just dump-ref-clefvis ref_clefvis_taps`), and every kernel of block 0
//! and of the merger against its host rule.
//!
//! The Clef file arm, per image of set B, in chain order (the first failing line names the earliest
//! op):
//! 1. Every tap the set holds (`inp_pos_emb`, blocks 0, 1, 13 and 26 of `ln1`, `Qcur_rope`,
//!    `attn_out`, `ffn_inp`, `ffn_out` and `layer_out`, `norm_b-27`, `embd`) against the reference's
//!    f32, as `rms(ours − ref) / rms(ref)`, within [`K_TAP`] times the larger of three controls of
//!    that tap: the reference's own CPU-twin distance from its CUDA run, the distance of the
//!    reference rounded to bf16 from itself (the floor of any bf16 tower), and, for a free-running
//!    tap, the network's sensitivity — our own encode against our encode with the last bit of one
//!    patch value in [`NOISE_EVERY`] flipped.
//! 2. Teacher-forced steps where the set holds a step's input: block 0 from `inp_pos_emb`, block 1
//!    from `layer_out-0`, the final norm and merger from `layer_out-26`, the merger from
//!    `norm_b-27`; the band is [`K_TAP`] times the larger of the rounding floor and the step's own
//!    sensitivity (the same step on its input with the noise of the sensitivity run).
//! 3. On the full-tap image of the set's smallest square grid, block 0 and the merger op by op
//!    against the host rules of `bloomery_gpu_vision`, each on our own tapped inputs: the position
//!    rows, the norms and the RoPE bit for bit (the RoPE table from the transcription of ggml's
//!    cache loop below, in merge order); every GEMM inside its accumulation-order bound; the
//!    attention inside its f32 bound; the residual epilogues bit for bit against a second,
//!    epilogue-free launch; the tanh GELU within one bf16 step of its host rule, the count of
//!    differing values pinned.
//! 4. Launches per image, two plain encodes bit-identical to each other and to the tapped run.
//!
//! Then once: the position rows of a 48×48 grid are the table itself; the V4.1 RoPE table is the
//! frozen loop's bytes; an encode does not depend on the larger image encoded before it in the
//! reused buffers; the bytes on the card equal the card figure.
//!
//! The Qwen3.6 and Qwen3.8 arms compare the final rows of each file with the sets of the
//! `qwen35moe` / `qwen4exp` vision families; a set that is not there ends its arm with a named
//! `deferred(<set>)` line and the run says how many arms it deferred. A deferral is not a pass.

#[cfg(not(feature = "vision"))]
fn main() {
    eprintln!(
        "gate_qwen3vl_tower: built without the `vision` feature; see `just gate-gpu-qwen3vl-tower`."
    );
    std::process::exit(2);
}

#[cfg(feature = "vision")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen3vl_tower", gate::run())
}

#[cfg(feature = "vision")]
#[path = "shared/vision_rules.rs"]
mod vision_rules;

#[cfg(feature = "vision")]
mod gate {
    use std::collections::{HashMap, HashSet};
    use std::path::Path;

    use super::vision_rules::{Count, exact_check, gemm_check, near_check, par_rows};
    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use bloomery_gpu_vision::attn::{QkvLayout, attn_72_ref, attn_bound};
    use bloomery_gpu_vision::chain::TapSink;
    use bloomery_gpu_vision::gelu_tanh::{GeluTanhKernels, gelu_tanh_ref};
    use bloomery_gpu_vision::gemm_bf16::{Epilogue, epilogue_ref, within_order};
    use bloomery_gpu_vision::layer_norm::layer_norm_ref;
    use bloomery_gpu_vision::pos_bilinear::{PosBilinearKernels, pos_bilinear_ref};
    use bloomery_gpu_vision::qwen3vl::{Encoder, scratch_bytes_of, weight_bytes_of};
    use bloomery_gpu_vision::rope2d::{HEAD_DIM_72, PAIRS_72, RopeTable, rope_72_ref};
    use bloomery_gpu_vision::{bf16_f32, f32_bf16};
    use cuda_core::DeviceBuffer;
    use gguf::Gguf;
    use refset::arch::qwen35::clefvis::{TAPS, TAPS_CPU_SET, TAPS_SET};
    use refset::clefvis::{ClefvisSet, MMPROJ_BF16, N_EMBD, TapScope};
    use vision::arch::qwen3vl::{Hparams, TokenLimits, preprocess};
    use vision::{Patches, Rgb8};

    const NAME: &str = "gate_qwen3vl_tower";
    /// The set's image whose block 0 and merger the op rules run on: the smallest square grid.
    const FULL: &str = "free-448x448";
    /// The set's image with the largest grid the token limit allows (`embd` only in the set).
    const LARGEST: &str = "max-2600x1800";
    /// Clef's card figures: the weights as uploaded, and the scratch at the default 4096-token
    /// limit (139,392 B per token).
    const CLEF_WEIGHT_BYTES: u64 = 924_123_136;
    const CLEF_SCRATCH_BYTES: u64 = 570_949_632;
    /// The blocks whose nodes the set holds.
    const TAP_BLOCKS: [usize; 4] = [0, 1, 13, 26];
    /// A tap passes when `rms(ours − ref) / rms(ref)` is at most this multiple of the larger of its
    /// controls.
    /// PIN(2026-10-10): fixed before the first run of this engine against the set; the design's
    /// prediction is a block output at about 1.6× the rounding floor, so 4 leaves room for a
    /// free-running network's amplification without admitting an order-one defect.
    const K_TAP: f64 = 4.0;
    /// The sensitivity run flips the last bit of every `NOISE_EVERY`-th patch value (a 1-ulp noise
    /// of the order of rounding the patches to bf16).
    const NOISE_EVERY: usize = 16;
    /// Values of the tanh GELU sweep of [−8, 8] that may differ from the host rule, each by one
    /// bf16 step at most: the device's `tanhf` against the host's.
    /// PIN(2026-10-10): the first measurement of this build; see the sweep line.
    const GELU_SWEEP_PIN: usize = 0;
    /// The same for the block-0 MLP's and the merger's rows of the full-tap image.
    /// PIN(2026-10-10): the first measurement of this build; see the rule lines.
    const GELU_ROWS_PIN: usize = 0;

    // ------------------------------------------------------------ distances

    #[derive(Clone, Copy)]
    struct Dist {
        max_rel: f64,
        rms_rel: f64,
    }

    /// `ours` (bf16 bits) against `refv` (f32): `max|Δ| / max|ref|` and `rms(Δ) / rms(ref)`.
    fn dist(ours: &[u16], refv: &[f32]) -> Dist {
        dist_f32(&ours.iter().map(|&b| bf16_f32(b)).collect::<Vec<_>>(), refv)
    }

    fn dist_f32(ours: &[f32], refv: &[f32]) -> Dist {
        let (mut max_d, mut max_r, mut sd, mut sr) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (&a, &b) in ours.iter().zip(refv) {
            let (x, y) = (f64::from(a), f64::from(b));
            max_d = max_d.max((x - y).abs());
            max_r = max_r.max(y.abs());
            sd += (x - y) * (x - y);
            sr += y * y;
        }
        Dist {
            max_rel: max_d / max_r,
            rms_rel: (sd / sr).sqrt(),
        }
    }

    /// The reference rounded to bf16 against itself: the floor of any bf16 tower's distance.
    fn floor(refv: &[f32]) -> f64 {
        let r: Vec<f32> = refv.iter().map(|&v| bf16_f32(f32_bf16(v))).collect();
        dist_f32(&r, refv).rms_rel
    }

    fn to_f32(v: &[u16]) -> Vec<f32> {
        v.iter().map(|&b| bf16_f32(b)).collect()
    }

    fn to_bf16(v: &[f32]) -> Vec<u16> {
        v.iter().map(|&x| f32_bf16(x)).collect()
    }

    /// One compared tap's line: the distance against `K_TAP ×` the controls.
    #[allow(
        clippy::too_many_arguments,
        reason = "one tap's names, values and three controls, as the line prints them"
    )]
    fn band_check(
        stem: &str,
        tap: &str,
        ours: &[u16],
        refv: &[f32],
        twin: f64,
        floor: f64,
        sens: Option<f64>,
    ) -> bool {
        if ours.len() != refv.len() {
            println!(
                "tap {stem:<14} {tap:<20} FAIL: {} values, the set has {}",
                ours.len(),
                refv.len()
            );
            return false;
        }
        let d = dist(ours, refv);
        let ctl = twin.max(floor).max(sens.unwrap_or(0.0));
        let pass = d.rms_rel <= K_TAP * ctl;
        println!(
            "tap {stem:<14} {tap:<20} max|rel| {:.3e}  rms rel {:.3e}  controls twin {twin:.2e} floor {floor:.2e} sens {}  ratio {:.2} (at most {K_TAP})  {}",
            d.max_rel,
            d.rms_rel,
            sens.map_or("-".to_string(), |s| format!("{s:.2e}")),
            d.rms_rel / ctl,
            verdict(pass)
        );
        pass
    }

    // ------------------------------------------------------------ taps

    /// Collects the named tensors of one encode.
    struct Collect {
        want: HashSet<String>,
        got: HashMap<String, Vec<u16>>,
    }

    impl TapSink for Collect {
        fn wants(&self, name: &str) -> bool {
            self.want.contains(name)
        }
        fn take(&mut self, name: &str, _cols: usize, bits: Vec<u16>) {
            self.got.insert(name.to_string(), bits);
        }
    }

    /// `(oracle node, our tap)` of every tap the set holds for blocks [`TAP_BLOCKS`], in chain
    /// order; the final rows (`embd`) are the encode's own output.
    fn oracle_taps() -> Vec<(String, String)> {
        let mut t = vec![("inp_pos_emb".to_string(), "embed".to_string())];
        for b in TAP_BLOCKS {
            for (node, op) in [
                ("ln1", "norm1"),
                ("Qcur_rope", "qrot"),
                ("attn_out", "attn"),
                ("ffn_inp", "resid1"),
                ("ffn_out", "mlp"),
            ] {
                t.push((format!("{node}-{b}"), format!("blk{b}.{op}")));
            }
            t.push((format!("layer_out-{b}"), format!("blk{b}")));
        }
        t.push(("norm_b-27".to_string(), "vit".to_string()));
        t
    }

    /// Everything our encode is asked to tap for an image: the oracle's taps, and for the
    /// full-tap image the block-0 and merger ops the host rules read.
    fn wanted(full: bool) -> HashSet<String> {
        let mut w: HashSet<String> = oracle_taps().into_iter().map(|(_, ours)| ours).collect();
        w.insert("pos".to_string());
        if full {
            for op in ["qkv", "krot", "sdpa", "norm2", "up", "act"] {
                w.insert(format!("blk0.{op}"));
            }
            for t in ["embed.patch", "merger.pre", "merger.h", "blk26"] {
                w.insert(t.to_string());
            }
            w.insert("blk26".to_string());
        }
        w
    }

    // ------------------------------------------------------------ independent host rules

    /// ggml's vision RoPE cache of one patch `(y, x)`, from the loop of
    /// `ggml_rope_cache_init` / `ggml_mrope_cache_init` (vision mode, sections of 18): pair `i`
    /// of the first section is the angle `y`, of the second `x`, each multiplied by `theta_scale`
    /// after every pair and started again at the section's own position.
    fn ggml_cache(y: usize, x: usize) -> Vec<f32> {
        let theta_scale = f64::from(10_000.0f32).powf(f64::from(-2.0f32 / 36.0f32)) as f32;
        let (mut theta_t, mut theta_h) = (y as f32, x as f32);
        let mut cache = Vec::with_capacity(2 * PAIRS_72);
        for pair in 0..PAIRS_72 {
            if pair == 18 {
                theta_h = x as f32;
            }
            let theta = if pair < 18 { theta_t } else { theta_h };
            cache.push(f64::from(theta).cos() as f32);
            cache.push(f64::from(theta).sin() as f32);
            theta_t *= theta_scale;
            theta_h *= theta_scale;
        }
        cache
    }

    /// The RoPE table of an `n_h × n_w` grid in merge order, row by row from [`ggml_cache`].
    fn ggml_table(n_h: usize, n_w: usize) -> Vec<f32> {
        let mut cs = Vec::with_capacity(n_h * n_w * 2 * PAIRS_72);
        for gy in 0..n_h / 2 {
            for gx in 0..n_w / 2 {
                for dy in 0..2 {
                    for dx in 0..2 {
                        cs.extend(ggml_cache(2 * gy + dy, 2 * gx + dx));
                    }
                }
            }
        }
        cs
    }

    /// V4.1's table builder as it was before the patch order became a parameter, frozen.
    fn frozen_v41_table(n_h: usize, n_w: usize, theta: f32) -> Vec<f32> {
        const PAIRS: usize = 32;
        const PER_AXIS: usize = PAIRS / 2;
        let freq: Vec<f32> = (0..PER_AXIS)
            .map(|i| {
                let e = (2 * i) as f32 / PAIRS as f32;
                let p = f64::from(theta).powf(f64::from(e)) as f32;
                1.0 / p
            })
            .collect();
        let mut cs = Vec::with_capacity(n_h * n_w * 2 * PAIRS);
        for h in 0..n_h {
            for w in 0..n_w {
                for (j, pos) in (0..PAIRS).map(|j| (j, if j < PER_AXIS { h } else { w })) {
                    let a = pos as f32 * freq[j % PER_AXIS];
                    cs.push(f64::from(a).cos() as f32);
                    cs.push(f64::from(a).sin() as f32);
                }
            }
        }
        cs
    }

    // ------------------------------------------------------------ host weights

    /// The host copies of the weights block 0 and the merger read, bf16 as the file holds them.
    struct HostW {
        pos: Vec<f32>,
        patch_w: Vec<u16>,
        patch_b: Vec<f32>,
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
        post_ln: (Vec<f32>, Vec<f32>),
        mm0_w: Vec<u16>,
        mm0_b: Vec<f32>,
        mm2_w: Vec<u16>,
        mm2_b: Vec<f32>,
    }

    fn host_w(file: &Gguf) -> Result<HostW, GateError> {
        use vision::arch::qwen3vl::names as n;
        let raw = |name: String| -> Result<&[u8], GateError> {
            let t = file
                .find(&name)
                .ok_or_else(|| format!("{name} not in the encoder file"))?;
            Ok(file.data(t)?)
        };
        let b16 = |name: String| -> Result<Vec<u16>, GateError> {
            let t = file
                .find(&name)
                .ok_or_else(|| format!("{name} not in the encoder file"))?;
            let bytes = file.data(t)?;
            if t.ty == gguf::GgmlType::F32 {
                Ok(bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32_bf16(f32::from_le_bytes(*c)))
                    .collect())
            } else {
                Ok(bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| u16::from_le_bytes(*c))
                    .collect())
            }
        };
        let f32s = |name: String| -> Result<Vec<f32>, GateError> {
            Ok(raw(name)?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        };
        let (first, second) = (
            b16(n::patch_embd_weight())?,
            b16(n::patch_embd_weight_second_frame())?,
        );
        let patch_w: Vec<u16> = first
            .as_chunks::<768>()
            .0
            .iter()
            .zip(second.as_chunks::<768>().0.iter())
            .flat_map(|(a, b)| a.iter().chain(b).copied())
            .collect();
        Ok(HostW {
            pos: f32s(n::position_embd())?,
            patch_w,
            patch_b: f32s(n::patch_embd_bias())?,
            ln1: (f32s(n::ln1_weight(0))?, f32s(n::ln1_bias(0))?),
            qkv_w: b16(n::attn_qkv_weight(0))?,
            qkv_b: f32s(n::attn_qkv_bias(0))?,
            o_w: b16(n::attn_out_weight(0))?,
            o_b: f32s(n::attn_out_bias(0))?,
            ln2: (f32s(n::ln2_weight(0))?, f32s(n::ln2_bias(0))?),
            up_w: b16(n::ffn_up_weight(0))?,
            up_b: f32s(n::ffn_up_bias(0))?,
            down_w: b16(n::ffn_down_weight(0))?,
            down_b: f32s(n::ffn_down_bias(0))?,
            post_ln: (f32s(n::post_ln_weight())?, f32s(n::post_ln_bias())?),
            mm0_w: b16(n::mm0_weight())?,
            mm0_b: f32s(n::mm0_bias())?,
            mm2_w: b16(n::mm2_weight())?,
            mm2_b: f32s(n::mm2_bias())?,
        })
    }

    fn norm_rows(x: &[u16], g: &(Vec<f32>, Vec<f32>), eps: f32) -> Vec<u16> {
        x.chunks_exact(g.0.len())
            .flat_map(|r| layer_norm_ref(r, &g.0, &g.1, eps))
            .collect()
    }

    // ------------------------------------------------------------ op rules

    /// Block 0 and the merger of the full-tap image, op by op, on our own tapped inputs.
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
        let (dim, ff, n) = (hp.dim, hp.ff, grid.0 * grid.1);
        let ff_pad = ff.next_multiple_of(64);
        let n_tok = n / 4;
        let merged = 4 * dim;
        let get = |k: &str| t.get(k).ok_or_else(|| format!("tap {k} was not taken"));
        let (pos, embed_patch, embed) = (get("pos")?, get("embed.patch")?, get("embed")?);
        let (norm1, qkv, qrot, krot) = (
            get("blk0.norm1")?,
            get("blk0.qkv")?,
            get("blk0.qrot")?,
            get("blk0.krot")?,
        );
        let (sdpa, attn_side, resid1, norm2) = (
            get("blk0.sdpa")?,
            get("blk0.attn")?,
            get("blk0.resid1")?,
            get("blk0.norm2")?,
        );
        let (up, act, mlp_side, blk0) = (
            get("blk0.up")?,
            get("blk0.act")?,
            get("blk0.mlp")?,
            get("blk0")?,
        );
        let (blk_last, vit, m_pre, m_h) = (
            get(&format!("blk{}", hp.n_layer - 1))?,
            get("vit")?,
            get("merger.pre")?,
            get("merger.h")?,
        );
        let k_patch = 2 * 3 * hp.patch * hp.patch;
        let mut ok = true;

        ok &= exact_check("pos rows", pos, &pos_bilinear_ref(&w.pos, 48, dim, grid));
        ok &= gemm_check(
            "patch_embed",
            patches,
            &w.patch_w,
            Some(&w.patch_b),
            n,
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
        ok &= exact_check("blk0.norm1", norm1, &norm_rows(embed, &w.ln1, hp.eps));
        ok &= gemm_check(
            "blk0.wqkv",
            norm1,
            &w.qkv_w,
            Some(&w.qkv_b),
            n,
            3 * dim,
            dim,
            qkv,
        );
        let table = vision_table(grid);
        let mut rot = qkv.clone();
        rope_72_ref(&mut rot, 3 * dim, 0, hp.n_head, &table);
        rope_72_ref(&mut rot, 3 * dim, dim, hp.n_head, &table);
        let cols = |x: &[u16], c0: usize| -> Vec<u16> {
            x.chunks_exact(3 * dim)
                .flat_map(|r| r[c0..c0 + dim].iter().copied())
                .collect()
        };
        ok &= exact_check("blk0.rope q", qrot, &cols(&rot, 0));
        ok &= exact_check("blk0.rope k", krot, &cols(&rot, dim));
        // The attention reads our own rotated q and k and the untouched v.
        let mut ours_rot = qkv.clone();
        for (i, row) in ours_rot.chunks_exact_mut(3 * dim).enumerate() {
            row[..dim].copy_from_slice(&qrot[i * dim..(i + 1) * dim]);
            row[dim..2 * dim].copy_from_slice(&krot[i * dim..(i + 1) * dim]);
        }
        let lay = QkvLayout {
            row_width: 3 * dim,
            q0: 0,
            k0: dim,
            v0: 2 * dim,
            n_heads: hp.n_head,
        };
        let scale = 1.0 / (HEAD_DIM_72 as f64).sqrt();
        let c = par_rows(n, |i| {
            let mut c = Count::default();
            for h in 0..hp.n_head {
                let (o, mag, logit) = attn_72_ref(&ours_rot, lay, n, scale, h, i);
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
        let want: Vec<u16> = attn_side
            .iter()
            .zip(embed)
            .map(|(&y, &r)| epilogue_ref(y, Epilogue::Residual, r))
            .collect();
        ok &= exact_check("blk0.resid1 epi", resid1, &want);
        ok &= exact_check("blk0.norm2", norm2, &norm_rows(resid1, &w.ln2, hp.eps));
        // The up projection: the file's 4304 columns by the exact product, the padded columns
        // exactly zero.
        let real: Vec<u16> = up
            .chunks_exact(ff_pad)
            .flat_map(|r| r[..ff].iter().copied())
            .collect();
        let pad_zero = up
            .chunks_exact(ff_pad)
            .all(|r| r[ff..].iter().all(|&v| bf16_f32(v) == 0.0));
        ok &= gemm_check("blk0.up", norm2, &w.up_w, Some(&w.up_b), n, ff, dim, &real);
        println!(
            "rule blk0.up padding    columns {ff}..{ff_pad} are zero: {}",
            verdict(pad_zero)
        );
        ok &= pad_zero;
        let real_act: Vec<u16> = act
            .chunks_exact(ff_pad)
            .flat_map(|r| r[..ff].iter().copied())
            .collect();
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
        let want: Vec<u16> = mlp_side
            .iter()
            .zip(resid1)
            .map(|(&y, &r)| epilogue_ref(y, Epilogue::Residual, r))
            .collect();
        ok &= exact_check("blk0 resid2 epi", blk0, &want);
        ok &= exact_check("post_ln", vit, &norm_rows(blk_last, &w.post_ln, hp.eps));
        ok &= gemm_check(
            "merger.mm0",
            vit,
            &w.mm0_w,
            Some(&w.mm0_b),
            n_tok,
            merged,
            merged,
            m_pre,
        );
        let want: Vec<u16> = m_pre.iter().map(|&y| gelu_tanh_ref(y)).collect();
        ok &= near_check("merger gelu_tanh", m_h, &want, GELU_ROWS_PIN);
        ok &= gemm_check(
            "merger.mm2",
            m_h,
            &w.mm2_w,
            Some(&w.mm2_b),
            n_tok,
            hp.out_dim,
            merged,
            out,
        );
        Ok(ok)
    }

    /// The host rule's RoPE table: [`ggml_table`] as the type the rule takes.
    fn vision_table(grid: (usize, usize)) -> bloomery_gpu_vision::rope2d::RopeTable72 {
        bloomery_gpu_vision::rope2d::RopeTable72 {
            n_h: grid.0,
            n_w: grid.1,
            cs: ggml_table(grid.0, grid.1),
        }
    }

    // ------------------------------------------------------------ once-per-run rules

    /// The tanh GELU on every normal bf16 value in [−8, 8] against the host rule.
    fn gelu_sweep(
        ctx: &std::sync::Arc<cuda_core::CudaContext>,
        stream: &cuda_core::CudaStream,
    ) -> Result<bool, GateError> {
        let xs: Vec<u16> = (0x0080..=0x4100u16).chain(0x8080..=0xC100).collect();
        let k = GeluTanhKernels::load(ctx)?;
        let mut d = DeviceBuffer::from_host(stream, &xs)?;
        k.enqueue(stream, xs.len(), &mut d)?;
        stream.synchronize()?;
        let got = d.to_host_vec(stream)?;
        let want: Vec<u16> = xs.iter().map(|&x| gelu_tanh_ref(x)).collect();
        Ok(near_check(
            "gelu_tanh sweep [-8, 8]",
            &got,
            &want,
            GELU_SWEEP_PIN,
        ))
    }

    /// The position rows of a 48×48 grid are the table itself in merge order (llama.cpp returns
    /// the table unresized when the grid is the table's), and a non-square grid matches the rule.
    fn pos_identity(
        ctx: &std::sync::Arc<cuda_core::CudaContext>,
        stream: &cuda_core::CudaStream,
        w: &HostW,
        dim: usize,
    ) -> Result<bool, GateError> {
        let k = PosBilinearKernels::load(ctx)?;
        let table = DeviceBuffer::from_host(stream, &w.pos)?;
        let mut ok = true;
        for grid in [(48usize, 48usize), (6, 10)] {
            let mut out = DeviceBuffer::<u16>::zeroed(stream, grid.0 * grid.1 * dim)?;
            k.enqueue(stream, &table, (48, dim), grid, &mut out)?;
            stream.synchronize()?;
            let got = out.to_host_vec(stream)?;
            ok &= exact_check(
                &format!("pos {}x{} host", grid.0, grid.1),
                &got,
                &pos_bilinear_ref(&w.pos, 48, dim, grid),
            );
            if grid == (48, 48) {
                let mut want = Vec::new();
                for gy in 0..24 {
                    for gx in 0..24 {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let p = (2 * gy + dy) * 48 + 2 * gx + dx;
                                want.extend(
                                    w.pos[p * dim..(p + 1) * dim].iter().map(|&v| f32_bf16(v)),
                                );
                            }
                        }
                    }
                }
                ok &= exact_check("pos 48x48 = table", &got, &want);
            }
        }
        Ok(ok)
    }

    /// V4.1's RoPE table builder, now taking the patch order, is the frozen loop's bytes.
    fn v41_table_bytes() -> bool {
        let mut ok = true;
        for (h, w) in [(1usize, 1usize), (4, 6), (23, 17), (64, 64)] {
            let new = RopeTable::new(h, w, 10_000.0);
            let old = frozen_v41_table(h, w, 10_000.0);
            ok &= new.cs.len() == old.len()
                && new
                    .cs
                    .iter()
                    .zip(&old)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
        }
        println!(
            "rule v4.1 rope table    RopeTable::new equals the frozen row-major loop bit for bit: {}",
            verdict(ok)
        );
        ok
    }

    // ------------------------------------------------------------ run

    struct Img {
        patches: Patches,
        grid: (usize, usize),
    }

    fn load_image(hp: &Hparams, name: &str) -> Result<Img, GateError> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/ref/clefvis/images");
        let bytes =
            std::fs::read(dir.join(format!("{name}.png"))).map_err(|e| format!("{name}: {e}"))?;
        let rule = hp.size_rule(TokenLimits::DEFAULT)?;
        let patches = preprocess(&Rgb8::from_png(&bytes)?, &rule)?;
        let grid = (patches.n_vit_h, patches.n_vit_w);
        Ok(Img { patches, grid })
    }

    pub fn run() -> Result<(), GateError> {
        let set = ClefvisSet::open(&TAPS.path(TAPS_SET), &TAPS)?;
        let twin = ClefvisSet::open(&TAPS.path(TAPS_CPU_SET), &TAPS)?;
        let file = Gguf::open(MMPROJ_BF16)?;
        let (ctx, stream) = bloomery_gpu::capsync::fresh_stream(0)?;
        let limits = TokenLimits::DEFAULT;
        let mut enc = Encoder::load(&ctx, &stream, &file, limits)?;
        let hp = enc.hparams().clone();
        let want_launches = Encoder::launches(hp.n_layer);
        let mut ok = true;
        println!(
            "load: {} blocks, dim {}, heads {}, ff {}, out {}; weights on the card {} B; activations {} B at {} tokens; {} launches per image",
            hp.n_layer,
            hp.dim,
            hp.n_head,
            hp.ff,
            hp.out_dim,
            enc.weight_bytes(),
            enc.scratch_bytes(),
            limits.max(),
            want_launches
        );
        let bytes_ok = enc.weight_bytes() == CLEF_WEIGHT_BYTES
            && enc.weight_bytes() == weight_bytes_of(&hp)
            && enc.scratch_bytes() == CLEF_SCRATCH_BYTES
            && enc.scratch_bytes() == scratch_bytes_of(&hp, limits.max())
            && hp.out_dim == N_EMBD;
        println!(
            "bytes: weights {} (want {CLEF_WEIGHT_BYTES}), scratch {} (want {CLEF_SCRATCH_BYTES}): {}",
            enc.weight_bytes(),
            enc.scratch_bytes(),
            verdict(bytes_ok)
        );
        ok &= bytes_ok;
        let hw = host_w(&file)?;
        ok &= gelu_sweep(&ctx, &stream)?;
        ok &= pos_identity(&ctx, &stream, &hw, hp.dim)?;
        ok &= v41_table_bytes();

        let mut names: Vec<(String, bool)> = set
            .images
            .iter()
            .map(|i| (i.name.clone(), i.name == FULL))
            .collect();
        names.sort_by_key(|(n, full)| (!*full, n.clone()));
        let oracle = oracle_taps();
        let mut full_state: Option<(Img, Vec<u16>)> = None;

        for (name, full) in &names {
            let img = load_image(&hp, name)?;
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
            let taps_full = set.tap_effect_of(name)?.scope == TapScope::Full;
            let mut sink = Collect {
                want: if taps_full {
                    wanted(*full)
                } else {
                    HashSet::new()
                },
                got: HashMap::new(),
            };
            let (out, tapped_launches) = {
                let r = enc.encode_tapped(&img.patches, &mut sink)?;
                stream.synchronize()?;
                (r.rows.buf().to_host_vec(&stream)?, r.launches)
            };

            // The sensitivity run: the same encode with one patch value in NOISE_EVERY nudged.
            let mut sens: HashMap<String, f64> = HashMap::new();
            if taps_full {
                let mut noisy = img.patches.clone();
                for (i, v) in noisy.bf16.iter_mut().enumerate() {
                    if i % NOISE_EVERY == 0 {
                        *v ^= 1;
                    }
                }
                let mut nsink = Collect {
                    want: sink.want.clone(),
                    got: HashMap::new(),
                };
                let nout = {
                    let r = enc.encode_tapped(&noisy, &mut nsink)?;
                    stream.synchronize()?;
                    r.rows.buf().to_host_vec(&stream)?
                };
                for (tap, ours) in &sink.got {
                    if let Some(n) = nsink.got.get(tap) {
                        let r: Vec<f32> = ours.iter().map(|&b| bf16_f32(b)).collect();
                        sens.insert(tap.clone(), dist(n, &r).rms_rel);
                    }
                }
                let r: Vec<f32> = out.iter().map(|&b| bf16_f32(b)).collect();
                sens.insert("embd".to_string(), dist(&nout, &r).rms_rel);
            }

            if taps_full {
                for (node, tap) in &oracle {
                    let ours = sink
                        .got
                        .get(tap)
                        .ok_or_else(|| format!("tap {tap} was not taken"))?;
                    let (_, refv) = set.tap(name, node)?;
                    let (_, twin_v) = twin.tap(name, node)?;
                    // Block 0's taps are within one block of the patch embedding, so the
                    // sensitivity control applies from block 1 on.
                    let free = !(tap == "embed" || tap.starts_with("blk0."));
                    ok &= band_check(
                        name,
                        tap,
                        ours,
                        &refv,
                        dist_f32(&twin_v, &refv).rms_rel,
                        floor(&refv),
                        free.then(|| sens[tap]),
                    );
                }
            }
            let (_, refv) = set.tap(name, "embd")?;
            let (_, twin_v) = twin.tap(name, "embd")?;
            ok &= band_check(
                name,
                "embd",
                &out,
                &refv,
                dist_f32(&twin_v, &refv).rms_rel,
                floor(&refv),
                sens.get("embd").copied(),
            );

            if taps_full {
                // Teacher-forced steps from the reference's own inputs. A step's control is the
                // larger of the rounding floor of its output and the step's own sensitivity: the
                // same step run on its input with one value in NOISE_EVERY nudged by an ulp. (The
                // twin's distance at a late tap measures the amplification over every block
                // before it, which a step fed the reference's own input does not carry.)
                let get =
                    |node: &str| -> Result<Vec<f32>, GateError> { Ok(set.tap(name, node)?.1) };
                let noisy = |x: &[u16]| -> Vec<u16> {
                    x.iter()
                        .enumerate()
                        .map(|(i, &v)| if i % NOISE_EVERY == 0 { v ^ 1 } else { v })
                        .collect()
                };
                for (b, input) in [(0usize, "inp_pos_emb"), (1, "layer_out-0")] {
                    let x = to_bf16(&get(input)?);
                    let got = enc.forced_block(img.grid, b, &x)?;
                    let sens =
                        dist(&enc.forced_block(img.grid, b, &noisy(&x))?, &to_f32(&got)).rms_rel;
                    let want = get(&format!("layer_out-{b}"))?;
                    ok &= band_check(
                        name,
                        &format!("forced.blk{b}"),
                        &got,
                        &want,
                        0.0,
                        floor(&want),
                        Some(sens),
                    );
                }
                let x = to_bf16(&get("layer_out-26")?);
                let (vit, rows) = enc.forced_tail(img.grid, &x, false)?;
                let (nvit, nrows) = enc.forced_tail(img.grid, &noisy(&x), false)?;
                let want = get("norm_b-27")?;
                ok &= band_check(
                    name,
                    "forced.vit",
                    &vit,
                    &want,
                    0.0,
                    floor(&want),
                    Some(dist(&nvit, &to_f32(&vit)).rms_rel),
                );
                let want = get("embd")?;
                ok &= band_check(
                    name,
                    "forced.tail embd",
                    &rows,
                    &want,
                    0.0,
                    floor(&want),
                    Some(dist(&nrows, &to_f32(&rows)).rms_rel),
                );
                let x = to_bf16(&get("norm_b-27")?);
                let (_, rows) = enc.forced_tail(img.grid, &x, true)?;
                let (_, nrows) = enc.forced_tail(img.grid, &noisy(&x), true)?;
                ok &= band_check(
                    name,
                    "forced.merger",
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
            let twice = va.0 == vb && va.0 == out;
            let launches = va.1 == want_launches && tapped_launches == want_launches;
            println!(
                "image {name}: eager twice bit-identical and equal to the tapped run: {}; launches {} (want {want_launches}): {}",
                verdict(twice),
                va.1,
                verdict(launches)
            );
            ok &= twice && launches;
            if *full {
                full_state = Some((img, out));
            }
        }

        // An encode does not depend on the larger image encoded before it in the reused buffers.
        if let Some((small, first)) = full_state {
            let largest = load_image(&hp, LARGEST)?;
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
            let pass = again == first && big.len() == tokens * hp.out_dim;
            println!(
                "reuse {FULL} after {LARGEST} ({tokens} of {} tokens): bit-identical to its first encode: {}",
                limits.max(),
                verdict(pass)
            );
            ok &= pass;
        }

        let deferred = deferred_arms();
        if ok {
            println!("{NAME}: PASS (clef arm; deferred={deferred})");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }

    /// The Qwen3.6 and Qwen3.8 B-final arms: the files' final rows against the sets of their
    /// families. A set that is not in place ends the arm with a named line, never a pass; returns
    /// how many arms did.
    fn deferred_arms() -> usize {
        let mut n = 0;
        for (arm, set) in [
            ("qwen3.6", "qwen35moe-vis-b"),
            ("qwen3.8", "qwen4exp-vis-b"),
        ] {
            println!("arm {arm}: deferred({set}): the family's set is not in this tree");
            n += 1;
        }
        n
    }
}
