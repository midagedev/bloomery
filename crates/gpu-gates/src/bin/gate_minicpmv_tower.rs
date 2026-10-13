//! GPU gate for the MiniCPM-V image tower on the card (`bloomery_gpu_vision::minicpmv`, the mmproj of
//! MiniCPM-V 4.7): the two kernels its vit_merger adds to the Qwen3-VL block set, each against its host
//! rule on seeded synthetic inputs, with the inputs on the card as the tower holds them (rows in merge
//! order, a 2×2 window being four consecutive rows); then the tower on the real file.
//!
//! 1. The windowed attention (`vis_attn_group`): 16 heads of 72 over fused `qkv` rows of 3456,
//!    swept over `n ∈ {4, 64, 1024}` rows in windows of four (and windows of 2 and 3, which the
//!    kernel takes as its `group` argument): every output value inside the f32 bound of
//!    `attn_bound` around the exact attention of its own window alone, the count of values over
//!    it pinned at [`ATTN_OUTSIDE_PIN`]; the canary words past the output untouched.
//! 2. The group mean (`vis_mean_groups`): bf16 rows of 1152 over the same sweep, equal to
//!    `mean_groups_ref` bit for bit, and a control that the inputs exercise the order — the count
//!    of values another association of the same sum rounds apart, after the bf16 rounding, must be
//!    above zero, or a gate on the order would pass on any order.
//! 3. A row count that is not a whole number of windows, a window of 0 rows and an empty image
//!    are refused by name by both entries, and launch nothing.
//! 4. The tower (real mmproj only): (a) the bytes on the card, the launches of a slice and the one
//!    narrowed tensor; (b) the vit_merger teacher-forced on a real image's block-6 output against
//!    its f32 host twin, and the mean on the merger's own stream bit for bit; (c) the final
//!    merger teacher-forced on the card's `vit` tap — norm, GEMMs and the erf GELU op by op, the
//!    whole against its twin; (d) three images (a 448×448 slice, a 1000×700 image that slices,
//!    200×200): finite rows, the rows each grid gives, eager twice, reuse after the largest.
//! 5. The tower against mainline's own nodes (real mmproj and the clefvis sets of
//!    `refset::arch::minicpmv4_7::vis`): for each tile of the three set images, fed the pixels
//!    mainline's preprocess produced (`inp_raw`), every tapped node (the patch embedding, blocks
//!    0, 6, 7 and 26, the vit_merger's stream, mean and output, the post-LN, the final merger's
//!    norm and rows) and the `embd` rows, which are the merger rows followed by the tile's
//!    separator rows (`tile_embd`), within a band derived from the set's CPU twin, the bf16 floor
//!    and the tower's sensitivity to a one-ulp nudge of its input; and that our plan and canvas
//!    are the set's.

#[cfg(not(feature = "vision"))]
fn main() {
    eprintln!(
        "gate_minicpmv_tower: built without the `vision` feature; see `just gate-gpu-minicpmv-tower`."
    );
    std::process::exit(2);
}

#[cfg(feature = "vision")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_minicpmv_tower", gate::run())
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
    use super::tower;
    use bloomery_gpu::GpuError;
    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use bloomery_gpu_vision::attn::{HEAD_DIM_72, QkvLayout, attn_bound};
    use bloomery_gpu_vision::attn_group::{AttnGroupArgs, AttnGroupKernels, attn_group_ref};
    use bloomery_gpu_vision::gemm_bf16::within_order;
    use bloomery_gpu_vision::mean_groups::{MeanGroupsKernels, mean_groups_ref};
    use bloomery_gpu_vision::{bf16_f32, f32_bf16};
    use cuda_core::{CudaStream, DeviceBuffer};

    const NAME: &str = "gate_minicpmv_tower";
    /// Heads of the merger's attention, and the width of a row they make.
    const HEADS: usize = 16;
    const DIM: usize = HEADS * HEAD_DIM_72;
    /// Width of a fused `qkv` row.
    const QKV_WIDTH: usize = 3 * DIM;
    /// `(rows, window)` of every sweep point: the tower's windows of four at three sizes, then
    /// windows the kernels take as an argument.
    const CASES: [(usize, usize); 5] = [(4, 4), (64, 4), (1024, 4), (64, 2), (66, 3)];
    /// Words past the end of an output buffer that the kernels must leave as they were.
    const CANARY: usize = 1024;
    /// A bf16 NaN: what the output holds before the launch, so a value never written is outside
    /// every bound.
    const NAN_BF16: u16 = 0x7fc0;
    /// Output values of the attention sweep that may lie outside the f32 bound.
    /// PIN(2026-10-13): the bound is derived (`attn_bound`), the kernel has no tensor-core
    /// split, so no value may lie outside it.
    const ATTN_OUTSIDE_PIN: usize = 0;

    /// A fixed xorshift64 stream.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        /// A value in `[0, 1)`.
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    // ------------------------------------------------------------ the windowed attention

    /// One sweep point of the attention: `n` rows in windows of `group`.
    fn attn_clause(
        stream: &CudaStream,
        k: &AttnGroupKernels,
        n: usize,
        group: usize,
    ) -> Result<bool, GateError> {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ ((n as u64) << 8) ^ group as u64);
        // Uniform in ±2.5: the scores of a row spread over a few units, so a window's weights
        // are neither uniform nor one-hot.
        let qkv: Vec<u16> = (0..n * QKV_WIDTH)
            .map(|_| f32_bf16(rng.unit() * 5.0 - 2.5))
            .collect();
        let lay = QkvLayout {
            row_width: QKV_WIDTH,
            q0: 0,
            k0: DIM,
            v0: 2 * DIM,
            n_heads: HEADS,
        };
        let d_qkv = DeviceBuffer::from_host(stream, &qkv)?;
        let mut d_out = DeviceBuffer::from_host(stream, &vec![NAN_BF16; n * DIM + CANARY])?;
        k.enqueue(
            stream,
            AttnGroupArgs {
                qkv: &d_qkv,
                layout: lay,
                n,
                group,
                scale: 1.0 / (HEAD_DIM_72 as f32).sqrt(),
                out_width: DIM,
                out: &mut d_out,
            },
        )?;
        stream.synchronize()?;
        let got = d_out.to_host_vec(stream)?;
        let scale = 1.0 / (HEAD_DIM_72 as f64).sqrt();
        let (mut outside, mut differ) = (0usize, 0usize);
        for i in 0..n {
            for h in 0..HEADS {
                let (o, mag, logit) = attn_group_ref(&qkv, lay, group, scale, h, i);
                for d in 0..HEAD_DIM_72 {
                    let y = got[i * DIM + h * HEAD_DIM_72 + d];
                    let b = attn_bound(o[d], mag[d], logit, group);
                    outside += usize::from(!within_order(y, o[d], b));
                    differ += usize::from(y != f32_bf16(o[d] as f32));
                }
            }
        }
        let canary = got[n * DIM..].iter().filter(|&&y| y != NAN_BF16).count();
        let pass = outside == ATTN_OUTSIDE_PIN && canary == 0;
        println!(
            "rule attn_group n {n:<4} window {group}: outside the f32 bound {outside} of {} (pin {ATTN_OUTSIDE_PIN}), differ from the exact rounding {differ} ({:.5}), canary words changed {canary}  {}",
            n * DIM,
            differ as f64 / (n * DIM) as f64,
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------------------ the group mean

    /// The designed inputs of one window of `group` rows of a mean sweep, as bf16 bit patterns:
    /// where two values of a window cancel and two are far below their ulp, the f32 adds round,
    /// so the order of the sum moves the result by more than a bf16 step. A window of four is
    /// `[B, s, −B, t]`, a window of three `[s, B, −B]` (the right fold and the left fold then
    /// differ), any other window plain values over sixteen binades.
    fn mean_window(rng: &mut Rng, group: usize) -> Vec<u16> {
        let e = (rng.unit() * 16.0) as i32 - 8;
        let big = (rng.unit() + 0.5) * ((e + 10) as f32).exp2();
        let tiny = |rng: &mut Rng, drop: i32| (rng.unit() - 0.5) * ((e - drop) as f32).exp2();
        let v = match group {
            4 => [big, tiny(rng, 11), -big, tiny(rng, 12)].to_vec(),
            3 => [tiny(rng, 11), big, -big].to_vec(),
            _ => (0..group)
                .map(|_| {
                    let e = (rng.unit() * 16.0) as i32 - 8;
                    (rng.unit() - 0.5) * (e as f32).exp2()
                })
                .collect(),
        };
        v.into_iter().map(f32_bf16).collect()
    }

    /// Another association of a window's sum, the control that the inputs exercise the order: the
    /// pairs `(p0 + p1) + (p2 + p3)` for a window of four, the right fold for a window of three,
    /// rounded to bf16 as the rule is. None for a window whose every order is the same sum.
    fn other_order(rows: &[u16], width: usize, group: usize, d: usize) -> Option<u16> {
        let p = |j: usize| bf16_f32(rows[j * width + d]);
        match group {
            4 => Some(f32_bf16(((p(0) + p(1)) + (p(2) + p(3))) * 0.25)),
            3 => Some(f32_bf16((p(0) + (p(1) + p(2))) * (1.0f32 / 3.0))),
            _ => None,
        }
    }

    /// One sweep point of the mean: `n` rows in windows of `group`.
    fn mean_clause(
        stream: &CudaStream,
        k: &MeanGroupsKernels,
        n: usize,
        group: usize,
    ) -> Result<bool, GateError> {
        let mut rng = Rng(0xd1b5_4a32_d192_ed03 ^ ((n as u64) << 8) ^ group as u64);
        // Each window's values are laid down column by column: value `j` of column `d` is row
        // `j` of the window.
        let n_out = n / group;
        let mut src = vec![0u16; n * DIM];
        for w in 0..n_out {
            for d in 0..DIM {
                for (j, v) in mean_window(&mut rng, group).into_iter().enumerate() {
                    src[(w * group + j) * DIM + d] = v;
                }
            }
        }
        let d_src = DeviceBuffer::from_host(stream, &src)?;
        let mut d_out = DeviceBuffer::from_host(stream, &vec![NAN_BF16; n_out * DIM + CANARY])?;
        k.enqueue(stream, n, group, DIM, &d_src, &mut d_out)?;
        stream.synchronize()?;
        let got = d_out.to_host_vec(stream)?;
        let want = mean_groups_ref(&src, DIM, group);
        let differ = got[..n_out * DIM]
            .iter()
            .zip(&want)
            .filter(|(a, b)| a != b)
            .count();
        let canary = got[n_out * DIM..]
            .iter()
            .filter(|&&v| v != NAN_BF16)
            .count();
        let pass = differ == 0 && canary == 0;
        println!(
            "rule mean_groups n {n:<4} window {group}: bit-exact: {differ} of {} differ, canary words changed {canary}  {}",
            n_out * DIM,
            verdict(pass)
        );
        // The control: another association of the same sums must round apart somewhere.
        let apart = src
            .chunks_exact(group * DIM)
            .flat_map(|g| (0..DIM).map(move |d| (g, d)))
            .zip(&want)
            .filter(|((g, d), w)| other_order(g, DIM, group, *d).is_some_and(|o| o != **w))
            .count();
        let applies = other_order(&src, DIM, group, 0).is_some();
        let control = !applies || apart > 0;
        println!(
            "control mean_groups n {n:<4} window {group}: another association rounds apart in {apart} of {} values{}  {}",
            n_out * DIM,
            if applies {
                ""
            } else {
                " (every order is the same sum: not applicable)"
            },
            verdict(control)
        );
        Ok(pass && control)
    }

    // ------------------------------------------------------------ the refusals

    /// `r` is the named refusal of `what` for a window that does not divide the rows.
    fn refused(name: &str, what: &str, r: Result<(), GpuError>) -> bool {
        let pass = matches!(
            &r,
            Err(GpuError::Shape { what: w, detail }) if *w == what && detail.contains("n % group must be 0")
        );
        println!(
            "refusal {name:<28} {}  {}",
            match &r {
                Ok(()) => "launched".to_string(),
                Err(e) => e.to_string(),
            },
            verdict(pass)
        );
        pass
    }

    /// Both entries refuse `n % group != 0`, a window of 0 and an empty image, before any launch;
    /// the stream then runs clean.
    fn refusals(
        stream: &CudaStream,
        a: &AttnGroupKernels,
        m: &MeanGroupsKernels,
    ) -> Result<bool, GateError> {
        let lay = QkvLayout {
            row_width: QKV_WIDTH,
            q0: 0,
            k0: DIM,
            v0: 2 * DIM,
            n_heads: HEADS,
        };
        // Buffers big enough for any of the shapes below, so only the window refuses.
        let rows = 8;
        let d_qkv = DeviceBuffer::<u16>::zeroed(stream, rows * QKV_WIDTH)?;
        let mut d_out = DeviceBuffer::<u16>::zeroed(stream, rows * DIM)?;
        let d_src = DeviceBuffer::<u16>::zeroed(stream, rows * DIM)?;
        let mut d_mean = DeviceBuffer::<u16>::zeroed(stream, rows * DIM)?;
        let mut ok = true;
        for (n, group) in [(6usize, 4usize), (4, 0), (0, 4)] {
            let r = a.enqueue(
                stream,
                AttnGroupArgs {
                    qkv: &d_qkv,
                    layout: lay,
                    n,
                    group,
                    scale: 1.0,
                    out_width: DIM,
                    out: &mut d_out,
                },
            );
            ok &= refused(
                &format!("attn_group n {n} window {group}"),
                "AttnGroupKernels::enqueue",
                r,
            );
            let r = m.enqueue(stream, n, group, DIM, &d_src, &mut d_mean);
            ok &= refused(
                &format!("mean_groups n {n} window {group}"),
                "MeanGroupsKernels::enqueue",
                r,
            );
        }
        stream.synchronize()?;
        Ok(ok)
    }

    // ------------------------------------------------------------ run

    pub fn run() -> Result<(), GateError> {
        let (ctx, stream) = bloomery_gpu::capsync::fresh_stream(0)?;
        let attn = AttnGroupKernels::load(&ctx)?;
        let mean = MeanGroupsKernels::load(&ctx)?;
        let mut ok = refusals(&stream, &attn, &mean)?;
        for (n, group) in CASES {
            ok &= attn_clause(&stream, &attn, n, group)?;
        }
        for (n, group) in CASES {
            ok &= mean_clause(&stream, &mean, n, group)?;
        }
        ok &= tower::clauses(&ctx, &stream)?;
        if ok {
            println!("{NAME}: PASS");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}

// ---------------------------------------------------------------- the tower on the real file

#[cfg(feature = "vision")]
mod tower {
    use std::collections::{HashMap, HashSet};
    use std::ops::AddAssign;
    use std::sync::Arc;

    use super::vision_rules::{exact_check, gemm_check, par_rows};
    use super::vision_taps::{
        Collect, HostFile, band_check, dist, dist_f32, eager_twice, floor, nudge, nudged,
        read_image, sens, to_f32,
    };
    use bloomery_gpu_gates::{GateError, verdict};
    use bloomery_gpu_vision::gelu_tanh::{gelu_erf_ref, gelu_tanh_ref};
    use bloomery_gpu_vision::gemm_bf16::{Epilogue, GemmArgs, GemmKernels, gelu_erf, within_order};
    use bloomery_gpu_vision::layer_norm::layer_norm_ref;
    use bloomery_gpu_vision::mean_groups::mean_groups_ref;
    use bloomery_gpu_vision::minicpmv::{
        Encoder, GROUP, HostFinalMerger, HostLinear, HostNorm, HostVitMerger, final_merger_f32,
        merge_rows, scratch_bytes_of, tile_embd, vit_merger_f32, weight_bytes_of, window_rows,
    };
    use bloomery_gpu_vision::{bf16_f32, f32_bf16};
    use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
    use gguf::Gguf;
    use refset::arch::minicpmv4_7::vis::{self, Kind, McvisSet};
    use refset::family::Family;
    use vision::arch::minicpmv::{Hparams, names, preprocess};
    use vision::{GridPlan, Patches, Rgb8};

    /// The checkpoint's mmproj.
    const MMPROJ: &str = "/models/MiniCPM-V-4.7-35B-A3B/mmproj-model-f16.gguf";
    /// Weights on the card: bf16 matrices with the MLP padded to the GEMM tile, the patch kernel
    /// narrowed, the position table and the separator rows on the host.
    const WEIGHT_BYTES: u64 = 1_101_559_552;
    /// A 448×448 slice: 1024 patches, 64 merged tokens.
    const SLICE_PATCHES: usize = 1024;
    /// Activations of a 448 slice (29,144 B a patch).
    const SLICE_SCRATCH_BYTES: u64 = 29_843_456;
    const BYTES_PER_PATCH: u64 = 29_144;
    /// Chain launches of one tile at 27 blocks.
    const LAUNCHES: usize = 230;
    /// Tiles of the 1000×700 image: 560×392 pixels, 40×28 patches.
    const SLICED_PATCHES: usize = 40 * 28;
    /// Merged tokens of a 448×448 tile and of a 560×392 tile.
    const TOKENS_448: usize = 64;
    const TOKENS_SLICED: usize = 70;
    /// The gate image (a 448×448 slice, the Qwen3-VL gate's), the image that slices, and the side
    /// of the small one cropped from the gate image's middle.
    const GATE_IMAGE: &str = "free-448x448";
    const SLICING_IMAGE: &str = "pad-1000x700";
    const SMALL_SIDE: usize = 200;
    /// A teacher-forced merger passes when `rms(card − twin) / rms(twin)` is at most this multiple
    /// of the larger of its two controls (see [`teacher_forced`]). The card rounds to bf16 at each
    /// op of the path, and independent roundings add in rms: the vit_merger's path has 11 (the
    /// norm, the qkv projection, the attention, the output projection, the stream's sum, the
    /// mean, the down-sample's norm, the up projection, the GELU, the down projection, the final
    /// sum), the final merger's 4 (the norm, the up projection, the GELU, the down projection),
    /// so the distance is about `√11 = 3.3` and `√4 = 2` floors; 6 is the 3.3 with a margin of
    /// 1.8 for the amplification of a norm or a cancelling sum.
    /// PIN(2026-10-13): the first measurement of this build, 3.59 and 2.87 floors.
    const K_TAP: f64 = 6.0;
    /// A value is inside the band when its distance from the twin is at most this multiple of the
    /// larger control times the largest of its own magnitude, the rms of its row and the rms of
    /// its channel.
    /// PIN(2026-10-13): the first measurement of this build: the farthest value lies at 29.7
    /// controls in the vit_merger and 19.0 in the final merger, so 48 keeps a margin of 1.6.
    const K_VAL: f64 = 48.0;
    /// Values of a merger's output outside that band.
    /// PIN(2026-10-13): the band is derived (see [`teacher_forced`]); no value may lie outside it.
    const OUTSIDE_PIN: usize = 0;
    /// The erf GELU's allowed distance from the host rule's f32 value, in units of `|x|`. The
    /// device runs the host's rule step for step (`z = x·√½`, `e = erf z`, `s = 1 + e`,
    /// `0.5·x·s`, each op rounded to f32), so the two differ by the `erf` alone and by how the
    /// roundings fall: the device's `erff` is within 2 ulp of the true value and the host's within
    /// 0.5, so `|Δe| ≤ 2.5·2⁻²⁴` (the ulp of a magnitude below 1); the two roundings of `s` add at
    /// most `2·2⁻²⁴` where `s ≥ 1` and `2·2⁻²⁵` below it; the two roundings of the product at
    /// most `2·2⁻²⁴` of its magnitude. For x > 0 that is `0.5·|x|·4.5·2⁻²⁴ + 2·2⁻²⁴·|x| =
    /// 2.125·2⁻²³·|x|`; for x < 0, where the sum `1 + e` cancels toward a small result and
    /// `s < 1`, `0.5·|x|·3.5·2⁻²⁴ + 2⁻²⁴·|x| = 1.375·2⁻²³·|x|`. `2⁻²¹ = 4·2⁻²³` covers both. The
    /// bound follows `|x|`, not the result: where `erf` is within an ulp of −1 the cancellation
    /// moves a small result by several bf16 steps of itself.
    const ERF_BOUND: f64 = 1.0 / 2_097_152.0;

    /// The twin's outputs of consecutive chunks of a run, with each chunk's place.
    #[derive(Default)]
    struct Chunks(Vec<(usize, Vec<f32>)>);

    impl AddAssign for Chunks {
        fn add_assign(&mut self, other: Chunks) {
            self.0.extend(other.0);
        }
    }

    /// A twin run on the host threads, window by window: both twins read four consecutive rows of
    /// `width` values as one window and no window reads another's rows (the twins' own tests hold
    /// that), so `f` on a chunk of whole windows gives that chunk of the whole run's output, bit
    /// for bit.
    fn by_windows(x: &[f32], width: usize, f: impl Fn(&[f32]) -> Vec<f32> + Sync) -> Vec<f32> {
        let windows = x.len() / (GROUP * width);
        let chunks = windows.clamp(1, 8);
        let per = windows.div_ceil(chunks);
        let mut parts = par_rows(chunks, |i| {
            let (a, b) = (i * per, ((i + 1) * per).min(windows));
            let rows = GROUP * width;
            if a >= b {
                return Chunks::default();
            }
            Chunks(vec![(i, f(&x[a * rows..b * rows]))])
        })
        .0;
        parts.sort_by_key(|p| p.0);
        parts.into_iter().flat_map(|p| p.1).collect()
    }

    fn tap<'a>(
        t: &'a std::collections::HashMap<String, Vec<u16>>,
        k: &str,
    ) -> Result<&'a Vec<u16>, GateError> {
        Ok(t.get(k).ok_or_else(|| format!("tap {k} was not taken"))?)
    }

    // ------------------------------------------------------------ host weights

    fn linear(
        hf: &HostFile<'_>,
        w: String,
        b: String,
        inp: usize,
    ) -> Result<HostLinear, GateError> {
        let w = to_f32(&hf.bf16(w)?);
        let b = hf.f32s(b)?;
        let out = b.len();
        if w.len() != inp * out {
            return Err(format!(
                "a linear layer of {out} outputs holds {} values, not {}",
                w.len(),
                inp * out
            )
            .into());
        }
        Ok(HostLinear { w, b, inp, out })
    }

    fn norm(hf: &HostFile<'_>, g: String, b: String) -> Result<HostNorm, GateError> {
        Ok(HostNorm {
            g: hf.f32s(g)?,
            b: hf.f32s(b)?,
        })
    }

    fn vit_merger_weights(hf: &HostFile<'_>, hp: &Hparams) -> Result<HostVitMerger, GateError> {
        let vm = |stem: &str| (names::vit_merger_weight(stem), names::vit_merger_bias(stem));
        let lin = |stem: &str, inp: usize| {
            let (w, b) = vm(stem);
            linear(hf, w, b, inp)
        };
        let (ln1, ds_ln) = (vm("ln1"), vm("ds_ln"));
        Ok(HostVitMerger {
            ln1: norm(hf, ln1.0, ln1.1)?,
            q: lin("attn_q", hp.dim)?,
            k: lin("attn_k", hp.dim)?,
            v: lin("attn_v", hp.dim)?,
            o: lin("attn_out", hp.dim)?,
            ds_ln: norm(hf, ds_ln.0, ds_ln.1)?,
            up: lin("ds_ffn_up", hp.merged_dim())?,
            down: lin("ds_ffn_down", 4 * hp.ff)?,
        })
    }

    struct FinalHost {
        merger: HostFinalMerger,
        up_w: Vec<u16>,
    }

    fn final_merger_weights(hf: &HostFile<'_>, hp: &Hparams) -> Result<FinalHost, GateError> {
        let merged = hp.merged_dim();
        Ok(FinalHost {
            merger: HostFinalMerger {
                norm: norm(
                    hf,
                    names::mm_weight("input_norm"),
                    names::mm_bias("input_norm"),
                )?,
                up: linear(hf, names::mm_weight("up"), names::mm_bias("up"), merged)?,
                down: linear(hf, names::mm_weight("down"), names::mm_bias("down"), merged)?,
            },
            up_w: hf.bf16(names::mm_weight("up"))?,
        })
    }

    // ------------------------------------------------------------ the band

    /// A merger's card rows `got` (bf16, rows of `cols`) against its twin `want` (f32, the same
    /// input), the twin run again on the input with the last bit of every 16th value flipped
    /// (`nudged_run`).
    ///
    /// The card is the twin plus the roundings it does: its stream, every op's output and the
    /// activation rows are bf16 (one rounding is a relative step of at most 2⁻⁹ of the value, rms
    /// about 2⁻⁸/√12), its GEMMs add f32 accumulation order (at most (k + 1)·2⁻²³ of Σ|terms| an
    /// output, the bound `gemm_check` holds each GEMM to) and its attention adds f32 softmax
    /// arithmetic (inside `attn_bound`). Both ride on the roundings: each is below the bf16 step
    /// of the op's output it lands in, so the band is drawn from two controls on the twin, as
    /// `gate_qwen3vl_tower` draws a teacher-forced step's. The *floor* F is the twin's own output
    /// rounded to bf16 against itself, `rms(bf16(t) − t) / rms(t)`: what one rounding of the
    /// output costs. The *sensitivity* S is `rms(t′ − t) / rms(t)` for the twin run on the nudged
    /// input t′: how much the merger amplifies a rounding of its input. The distance passes when
    /// `rms(card − twin) / rms(twin) ≤ K_TAP · max(F, S)`. Every value must also be within
    /// `K_VAL · max(F, S) · max(|t_i|, rms of its row, rms of its channel)` of its twin, the count
    /// outside pinned at [`OUTSIDE_PIN`]: the roundings that land in a value (the last one, the
    /// residual's) follow the value; the rest follow the scale of the activations of its row and
    /// the gain of its output channel (the weights of a channel with a large rms are large), not
    /// the result they sum to, because a GEMM's error follows its terms.
    fn teacher_forced(
        name: &str,
        got: &[u16],
        want: &[f32],
        nudged_run: &[f32],
        cols: usize,
    ) -> bool {
        let got_f = to_f32(got);
        let d = dist_f32(&got_f, want);
        let (f, s) = (floor(want), dist_f32(nudged_run, want).rms_rel);
        let ctl = f.max(s);
        let (mut outside, mut worst) = (0usize, 0.0f64);
        let rows = want.len() / cols;
        let chan: Vec<f64> = (0..cols)
            .map(|c| {
                ((0..rows)
                    .map(|r| f64::from(want[r * cols + c]).powi(2))
                    .sum::<f64>()
                    / rows as f64)
                    .sqrt()
            })
            .collect();
        for (y, t) in got_f.chunks_exact(cols).zip(want.chunks_exact(cols)) {
            let row_rms =
                (t.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / cols as f64).sqrt();
            for (c, (&y, &t)) in y.iter().zip(t).enumerate() {
                let e = (f64::from(y) - f64::from(t)).abs();
                let scale = ctl * f64::from(t).abs().max(row_rms).max(chan[c]);
                outside += usize::from(e > K_VAL * scale);
                worst = worst.max(e / scale);
            }
        }
        let ratio = d.rms_rel / ctl;
        let pass = got.len() == want.len() && ratio <= K_TAP && outside == OUTSIDE_PIN;
        println!(
            "teacher-forced {name:<14} rms rel {:.3e} = {ratio:.2} x control {ctl:.3e} (floor {f:.3e}, sensitivity {s:.3e}; at most {K_TAP}); {outside} of {} values outside {K_VAL} controls (pin {OUTSIDE_PIN}), the farthest at {worst:.2}  {}",
            d.rms_rel,
            want.len(),
            verdict(pass)
        );
        pass
    }

    /// The erf GELU's results `got` for the bf16 inputs `xs` against the host rule inside
    /// [`ERF_BOUND`]: how many are not a rounding of a value within `|x|·ERF_BOUND` of the rule's
    /// f32 value, and how many differ from the rule's own rounding.
    fn erf_outside(xs: &[u16], got: &[u16]) -> (usize, usize) {
        let (mut outside, mut differ) = (0usize, 0usize);
        for (&x, &y) in xs.iter().zip(got) {
            let xf = bf16_f32(x);
            let centre = f64::from(gelu_erf(xf));
            outside += usize::from(!within_order(y, centre, f64::from(xf).abs() * ERF_BOUND));
            differ += usize::from(y != gelu_erf_ref(x));
        }
        (outside, differ)
    }

    // ------------------------------------------------------------ images

    fn small_image() -> Result<Rgb8, GateError> {
        let src = read_image(GATE_IMAGE)?;
        let (x0, y0) = ((src.width - SMALL_SIDE) / 2, (src.height - SMALL_SIDE) / 2);
        let mut data = Vec::with_capacity(SMALL_SIDE * SMALL_SIDE * 3);
        for y in y0..y0 + SMALL_SIDE {
            data.extend_from_slice(
                &src.data[(y * src.width + x0) * 3..(y * src.width + x0 + SMALL_SIDE) * 3],
            );
        }
        Ok(Rgb8 {
            width: SMALL_SIDE,
            height: SMALL_SIDE,
            data,
        })
    }

    /// The rows of every tile of `tiles` on the card, in tile order.
    fn encode_tiles(
        enc: &mut Encoder,
        stream: &CudaStream,
        tiles: &[Patches],
    ) -> Result<Vec<(Vec<u16>, usize)>, GateError> {
        let mut all = Vec::new();
        for t in tiles {
            let r = enc.encode(t)?;
            all.push((r.rows.buf().to_host_vec(stream)?, r.launches));
        }
        Ok(all)
    }

    fn finite(rows: &[u16]) -> bool {
        rows.iter().all(|&v| v & 0x7f80 != 0x7f80)
    }

    // ------------------------------------------------------------ mainline's nodes

    /// A tap within one block of the input (the patch embedding, block 0) passes against mainline
    /// when `rms(ours − ref) / rms(ref)` is at most `K_MARGIN · √N` times the larger of its controls
    /// (the reference's own CPU-twin distance and the bf16 floor; see [`NODES`] for `N`, the count
    /// of independent bf16 roundings that reach the tap). Independent roundings add in rms, so the
    /// distance is about `√N` floors; the margin covers the amplification of a norm or a cancelling
    /// sum.
    /// PIN(2026-10-13): fixed before the first run of the tower against these sets; the margin
    /// `K_TAP` leaves over its `√11` floors (6 over 3.3). The first measurement of this build, over
    /// the seven tiles: ratios of 1.2 to 1.3 at the patch embedding and 2.3 to 2.5 at block 0.
    const K_MARGIN: f64 = 1.8;

    /// A tap from block 1 on passes when the distance is at most `K_SENS` times the larger of its
    /// controls, the sensitivity to a nudge of the input among them. The nudge runs through the
    /// same blocks the distance does, so it already carries their amplification: the `√N` of
    /// [`K_MARGIN`] would count it twice.
    /// PIN(2026-10-13): the first measurement of this build, over the seven tiles: ratios of 0.8 to
    /// 1.3 from block 6 on, so 3 leaves a margin of 2.3; the merge-order mutant reads 15.4 at its
    /// least (the vit_merger's attention), 5 times this band.
    const K_SENS: f64 = 3.0;

    /// Our name for the final merger's rows, which the encode returns and no tap carries.
    const MERGER_OUT: &str = "merger_out";

    /// The roundings that reach the final merger's rows and `embd`: the 62 of the post-LN, the
    /// final norm, and the up projection, the GELU and the down projection.
    const N_FINAL: usize = 66;

    /// Where a tap's rows lie on mainline's side against ours.
    #[derive(Clone, Copy)]
    enum Rows {
        /// One row a patch before the down-sample: mainline's in raster order, ours in merge order
        /// (`merge_rows`).
        Raster,
        /// One row a 2×2 window after it: mainline's in the window raster order of its index
        /// tensors, ours four consecutive merge-order rows a window (`window_rows`).
        Window,
        /// One row a group of the final merger, in the same order on both sides.
        Group,
    }

    /// `(mainline's node, our tap, the order of its rows, N)` of every node the sets hold that the
    /// tower can tap, in chain order. `N` counts the independent bf16 roundings that reach the tap
    /// with a weight of about 1: the patch embedding's 4 (the pixels, the f16 weights, the position
    /// rows, the output); 2 for each block (the attention's and the MLP's sum into the stream; the
    /// roundings inside a branch ride on the margin); 1 each for the vit_merger's attention sum,
    /// its mean and its output sum; then the post-LN's 1, the final norm's 1 and [`N_FINAL`].
    const NODES: [(&str, &str, Rows, usize); 11] = [
        ("pos_embed", "embed", Rows::Raster, 4),
        ("layer_out-0", "blk0", Rows::Raster, 6),
        ("layer_out-6", "blk6", Rows::Raster, 18),
        (
            "vit_merger_attn_residual",
            "vit_merger.attn",
            Rows::Raster,
            19,
        ),
        (
            "vit_merger_ds_mean_res",
            "vit_merger.mean",
            Rows::Window,
            20,
        ),
        ("vit_merger_ds_out", "vit_merger", Rows::Window, 21),
        ("layer_out-7", "blk7", Rows::Window, 23),
        ("layer_out-26", "blk26", Rows::Window, 61),
        ("post_ln", "vit", Rows::Window, 62),
        ("merger_normed", "merger.norm", Rows::Group, 63),
        ("merger_out", MERGER_OUT, Rows::Group, N_FINAL),
    ];

    /// The card's set, its CPU twin and the preprocess set.
    struct Oracle {
        card: McvisSet,
        twin: McvisSet,
        pre: McvisSet,
    }

    /// `rows` of `width` values in the order `order` names: row `r` of the result is row `order[r]`
    /// of the input.
    fn reorder(rows: &[f32], width: usize, order: &[usize]) -> Vec<f32> {
        order
            .iter()
            .flat_map(|&i| rows[i * width..(i + 1) * width].iter().copied())
            .collect()
    }

    /// A tile's patches from mainline's own pixels (`inp_raw`: planar `[W, H, 3]` f32): the patch
    /// grid in raster order, each patch `3 × 14 × 14` values (channel, row, column), rounded to
    /// bf16, which is what the tower takes.
    fn mainline_patches(inp: &[f32], t: &vis::Tile) -> Result<Patches, GateError> {
        let (w, h, p) = (t.w, t.h, vis::PATCH);
        if inp.len() != 3 * w * h {
            return Err(format!(
                "tile {}: inp_raw holds {} values, the tile is {w}x{h}x3",
                t.index,
                inp.len()
            )
            .into());
        }
        let (n_vit_h, n_vit_w) = (h / p, w / p);
        let mut bf16 = Vec::with_capacity(n_vit_h * n_vit_w * 3 * p * p);
        for i in 0..n_vit_h {
            for j in 0..n_vit_w {
                for c in 0..3 {
                    for py in 0..p {
                        let at = (c * h + i * p + py) * w + j * p;
                        bf16.extend(inp[at..at + p].iter().map(|&v| f32_bf16(v)));
                    }
                }
            }
        }
        Ok(Patches {
            plan: GridPlan {
                n_llm_h: t.ny,
                n_llm_w: t.nx,
                best_h: h,
                best_w: w,
            },
            n_vit_h,
            n_vit_w,
            patch_len: 3 * p * p,
            bf16,
        })
    }

    /// A tile's separator rows as the table's row numbers.
    fn suffix_of(t: &vis::Tile) -> Result<Vec<u8>, GateError> {
        Ok(t.suffix
            .iter()
            .map(|&s| u8::try_from(s))
            .collect::<Result<_, _>>()
            .map_err(|e| format!("tile {}: a separator row: {e}", t.index))?)
    }

    /// Our plan and canvas for the image's PNG against the set's `# image` and `# tile` lines (the
    /// tile sizes, the token grids, the separator rows, the rows of each tile and of the image),
    /// and, printed without a verdict, how far our preprocess's bf16 patches lie from mainline's
    /// own pixels rounded the same way.
    fn preproc_clause(
        hp: &Hparams,
        name: &str,
        img: &vis::Image,
        mainline: &[Patches],
    ) -> Result<bool, GateError> {
        let g = preprocess(&read_image(name)?, hp)?;
        let canvas = g.canvas();
        let mut same = g.tiles.len() == img.tiles.len()
            && canvas.tiles() == img.tiles.len()
            && canvas.rows() == img.n_tokens
            && img.n_tokens == img.tiles.iter().map(|t| t.rows).sum::<usize>();
        for (k, (t, ours)) in img.tiles.iter().zip(&g.tiles).enumerate() {
            let c = canvas.tile(k);
            same &= (ours.plan.best_w, ours.plan.best_h) == (t.w, t.h)
                && (c.nx, c.ny) == (t.nx, t.ny)
                && c.seps == suffix_of(t)?.as_slice()
                && t.merger_rows == t.nx * t.ny
                && t.rows == t.merger_rows + t.suffix.len();
        }
        println!(
            "preproc {name}: {} tiles, {} rows; our plan, canvas and separator rows equal the set's  {}",
            img.tiles.len(),
            img.n_tokens,
            verdict(same)
        );
        for (t, (ours, theirs)) in img.tiles.iter().zip(g.tiles.iter().zip(mainline)) {
            let differ = ours
                .bf16
                .iter()
                .zip(&theirs.bf16)
                .filter(|(a, b)| a != b)
                .count();
            println!(
                "preproc {name}/t{}: our patches differ from mainline's pixels in {differ} of {} bf16 values, rms rel {:.2e} (printed, no verdict)",
                t.index,
                theirs.bf16.len(),
                dist(&ours.bf16, &to_f32(&theirs.bf16)).rms_rel
            );
        }
        Ok(same)
    }

    /// [`band_check`] for ours in f32: `embd` holds the table's separator rows, which are not bf16
    /// values.
    #[allow(
        clippy::too_many_arguments,
        reason = "one tap's names, values and three controls, as the line prints them"
    )]
    fn band_f32(
        k: f64,
        stem: &str,
        label: &str,
        ours: &[f32],
        refv: &[f32],
        twin: f64,
        floor: f64,
        sens: f64,
    ) -> bool {
        if ours.len() != refv.len() {
            println!(
                "tap {stem:<14} {label:<20} FAIL: {} values, the set has {}",
                ours.len(),
                refv.len()
            );
            return false;
        }
        let d = dist_f32(ours, refv);
        let ctl = twin.max(floor).max(sens);
        let pass = d.rms_rel <= k * ctl;
        println!(
            "tap {stem:<14} {label:<20} max|rel| {:.3e}  rms rel {:.3e}  controls twin {twin:.2e} floor {floor:.2e} sens {sens:.2e}  ratio {:.2} (at most {k})  {}",
            d.max_rel,
            d.rms_rel,
            d.rms_rel / ctl,
            verdict(pass)
        );
        pass
    }

    /// One encode of `patches` with the taps `want` names, and the final rows as [`MERGER_OUT`].
    fn tapped_encode(
        enc: &mut Encoder,
        stream: &CudaStream,
        patches: &Patches,
        want: &HashSet<String>,
    ) -> Result<HashMap<String, Vec<u16>>, GateError> {
        let mut sink = Collect::new(want.clone());
        let out = {
            let r = enc.encode_tapped(patches, &mut sink)?;
            r.rows.buf().to_host_vec(stream)?
        };
        let mut got = sink.got;
        got.insert(MERGER_OUT.to_string(), out);
        Ok(got)
    }

    /// Every node of tile `t` of image `name` against the card's set, from mainline's own pixels,
    /// then its `embd`: the merger rows with the tile's separator rows after them, in a band, and
    /// the separator rows equal to mainline's bit for bit.
    fn tile_clauses(
        enc: &mut Encoder,
        stream: &CudaStream,
        hp: &Hparams,
        oracle: &Oracle,
        name: &str,
        t: &vis::Tile,
        patches: &Patches,
    ) -> Result<bool, GateError> {
        let want: HashSet<String> = NODES
            .iter()
            .filter(|n| n.1 != MERGER_OUT)
            .map(|n| n.1.to_string())
            .collect();
        let ours = tapped_encode(enc, stream, patches, &want)?;
        let mut noisy = patches.clone();
        nudge(&mut noisy.bf16);
        let nudged_run = tapped_encode(enc, stream, &noisy, &want)?;
        let stem = format!("{name}/t{}", t.index);
        let (n_h, n_w) = (patches.n_vit_h, patches.n_vit_w);
        let (raster, window) = (merge_rows(n_h, n_w), window_rows(n_h, n_w));
        let mut ok = true;
        for (node, label, order, n) in NODES {
            let x = ours
                .get(label)
                .ok_or_else(|| format!("tap {label} was not taken"))?;
            let (_, refv) = oracle.card.tap(name, t.index, node)?;
            let (_, twin_v) = oracle.twin.tap(name, t.index, node)?;
            let rows = match order {
                Rows::Raster => n_h * n_w,
                Rows::Window => n_h * n_w / GROUP,
                Rows::Group => t.merger_rows,
            };
            let want_v = match order {
                Rows::Raster => reorder(&refv, refv.len() / rows, &raster),
                Rows::Window => reorder(&refv, refv.len() / rows, &window),
                Rows::Group => refv.clone(),
            };
            // Block 0 and the patch embedding are within one block of the input, where a nudge
            // measures the input's rounding and no amplification.
            let free = !matches!(label, "embed" | "blk0");
            ok &= band_check(
                if free {
                    K_SENS
                } else {
                    K_MARGIN * (n as f64).sqrt()
                },
                &stem,
                label,
                x,
                &want_v,
                dist_f32(&twin_v, &refv).rms_rel,
                floor(&want_v),
                free.then(|| sens(&nudged_run[label], x)),
            );
        }
        let (width, split) = (hp.out_dim, t.merger_rows * hp.out_dim);
        let seps = suffix_of(t)?;
        let table = enc.separators();
        let embd = tile_embd(&to_f32(&ours[MERGER_OUT]), table, width, &seps)?;
        let embd_nudged = tile_embd(&to_f32(&nudged_run[MERGER_OUT]), table, width, &seps)?;
        let (_, ref_embd) = oracle.card.tap(name, t.index, vis::EMBD)?;
        let (_, twin_embd) = oracle.twin.tap(name, t.index, vis::EMBD)?;
        ok &= band_f32(
            K_SENS,
            &stem,
            "embd",
            &embd,
            &ref_embd,
            dist_f32(&twin_embd, &ref_embd).rms_rel,
            floor(&ref_embd),
            dist_f32(&embd_nudged, &embd).rms_rel,
        );
        let tail_equal = match (embd.get(split..), ref_embd.get(split..)) {
            (Some(a), Some(b)) => {
                embd.len() == t.rows * width
                    && a.len() == b.len()
                    && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
            }
            _ => false,
        };
        println!(
            "tap {stem:<14} {:<20} the {} separator rows {seps:?} equal mainline's, bit for bit  {}",
            "embd.sep",
            seps.len(),
            verdict(tail_equal)
        );
        Ok(ok && tail_equal)
    }

    /// The tower against the sets of mainline's nodes, tile by tile of every image the sets hold.
    fn mainline_clauses(
        enc: &mut Encoder,
        stream: &CudaStream,
        hp: &Hparams,
    ) -> Result<bool, GateError> {
        let open = |f: &Family, i: usize, kind| McvisSet::open(&f.path(f.sets[i]), kind);
        let oracle = Oracle {
            card: open(&vis::TAPS, 0, Kind::Taps)?,
            twin: open(&vis::TAPS, 1, Kind::Taps)?,
            pre: open(&vis::PREPROC, 0, Kind::Preproc)?,
        };
        let geometry = (hp.n_layer, hp.merger_after, hp.dim, hp.out_dim)
            == (vis::BLOCKS, vis::INSERT_LAYER, vis::VIT_WIDTH, vis::N_EMBD)
            && vis::TAP_BLOCKS == [0, hp.merger_after, hp.merger_after + 1, hp.n_layer - 1];
        println!(
            "mainline sets: {} images; the card's dumped on {}, its twin on {}; blocks, insert layer and widths equal the nodes' table: {}",
            oracle.card.images.len(),
            oracle.card.card,
            oracle.twin.card,
            verdict(geometry)
        );
        let mut ok = geometry;
        for img in &oracle.card.images {
            let mut mainline = Vec::new();
            for t in &img.tiles {
                mainline.push(mainline_patches(
                    &oracle.pre.inp_raw(&img.name, t.index)?,
                    t,
                )?);
            }
            ok &= preproc_clause(hp, &img.name, img, &mainline)?;
            for (t, patches) in img.tiles.iter().zip(&mainline) {
                ok &= tile_clauses(enc, stream, hp, &oracle, &img.name, t, patches)?;
            }
        }
        Ok(ok)
    }

    // ------------------------------------------------------------ the clauses

    #[allow(clippy::too_many_lines, reason = "the tower's clauses in chain order")]
    pub fn clauses(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>) -> Result<bool, GateError> {
        let file = Gguf::open(MMPROJ)?;
        let hp = Hparams::read(&file).map_err(|e| format!("{MMPROJ}: {e}"))?;
        let gate = preprocess(&read_image(GATE_IMAGE)?, &hp)?;
        let sliced = preprocess(&read_image(SLICING_IMAGE)?, &hp)?;
        let small = preprocess(&small_image()?, &hp)?;
        let widest = [&gate, &sliced, &small]
            .iter()
            .flat_map(|g| g.tiles.iter().map(|t| t.n_vit_h * t.n_vit_w))
            .max()
            .unwrap_or(0);
        let mut enc = Encoder::load(ctx, stream, &file, widest)?;
        let mut ok = true;

        // (a) load and card facts
        println!("{}", enc.describe());
        let narrowed = names::patch_embd_weight();
        let facts = enc.weight_bytes() == WEIGHT_BYTES
            && enc.weight_bytes() == weight_bytes_of(&hp)
            && scratch_bytes_of(&hp, SLICE_PATCHES) == SLICE_SCRATCH_BYTES
            && widest == SLICED_PATCHES
            && enc.scratch_bytes() == scratch_bytes_of(&hp, widest)
            && enc.scratch_bytes() == widest as u64 * BYTES_PER_PATCH
            && Encoder::launches(hp.n_layer) == LAUNCHES
            && enc.narrowed() == [narrowed.as_str()];
        println!(
            "bytes: weights {} (want {WEIGHT_BYTES}), a 448 slice's scratch {} (want {SLICE_SCRATCH_BYTES}), this run's tile bound {widest} patches {} B; launches {} (want {LAUNCHES}); narrowed {:?}: {}",
            enc.weight_bytes(),
            scratch_bytes_of(&hp, SLICE_PATCHES),
            enc.scratch_bytes(),
            Encoder::launches(hp.n_layer),
            enc.narrowed(),
            verdict(facts)
        );
        ok &= facts;

        // (b) and (c): one real image's taps
        let dim = hp.dim;
        let tile = &gate.tiles[0];
        let n = tile.n_vit_h * tile.n_vit_w;
        let blk = format!("blk{}", hp.merger_after);
        let want: HashSet<String> = [
            blk.as_str(),
            "vit_merger.attn",
            "vit_merger.mean",
            "vit_merger",
            "vit",
            "merger.norm",
            "merger.h",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        let mut sink = Collect::new(want);
        let (tapped, tapped_launches) = {
            let r = enc.encode_tapped(tile, &mut sink)?;
            (r.rows.buf().to_host_vec(stream)?, r.launches)
        };
        let t = &sink.got;
        let hf = HostFile(&file);

        let merger = vit_merger_weights(&hf, &hp)?;
        let x = to_f32(tap(t, &blk)?);
        let twin = by_windows(&x, dim, |c| vit_merger_f32(&merger, c, hp.eps));
        let x_nudged = to_f32(&nudged(tap(t, &blk)?));
        let twin_nudged = by_windows(&x_nudged, dim, |c| vit_merger_f32(&merger, c, hp.eps));
        ok &= teacher_forced(
            "vit_merger",
            tap(t, "vit_merger")?,
            &twin,
            &twin_nudged,
            dim,
        );
        ok &= exact_check(
            "vit_merger mean",
            tap(t, "vit_merger.mean")?,
            &mean_groups_ref(tap(t, "vit_merger.attn")?, dim, GROUP),
        );

        let merged = hp.merged_dim();
        let n_tok = n / (GROUP * GROUP);
        let last = final_merger_weights(&hf, &hp)?;
        let vit = tap(t, "vit")?;
        let ln: Vec<u16> = vit
            .chunks_exact(merged)
            .flat_map(|r| layer_norm_ref(r, &last.merger.norm.g, &last.merger.norm.b, hp.eps))
            .collect();
        ok &= exact_check("merger.norm", tap(t, "merger.norm")?, &ln);
        // The up layer's GEMM with no epilogue, then the fused one's GELU against the erf rule.
        let gemm = GemmKernels::load(ctx, stream)?;
        let d_norm = DeviceBuffer::from_host(stream, tap(t, "merger.norm")?)?;
        let d_up_w = DeviceBuffer::from_host(stream, &last.up_w)?;
        let d_up_b = DeviceBuffer::from_host(stream, &last.merger.up.b)?;
        let mut d_pre = DeviceBuffer::<u16>::zeroed(stream, n_tok * merged)?;
        gemm.enqueue(
            stream,
            GemmArgs {
                a: &d_norm,
                b: &d_up_w,
                bias: Some(&d_up_b),
                epilogue: Epilogue::None,
                resid: None,
                m: n_tok,
                n: merged,
                k: merged,
                c: &mut d_pre,
            },
        )?;
        stream.synchronize()?;
        let pre = d_pre.to_host_vec(stream)?;
        ok &= gemm_check(
            "merger.up",
            tap(t, "merger.norm")?,
            &last.up_w,
            Some(&last.merger.up.b),
            n_tok,
            merged,
            merged,
            &pre,
        );
        let h = tap(t, "merger.h")?;
        let (outside, differ) = erf_outside(&pre, h);
        let tells = pre
            .iter()
            .filter(|&&x| {
                let xf = bf16_f32(x);
                let tanh = gelu_tanh_ref(x);
                !within_order(
                    tanh,
                    f64::from(gelu_erf(xf)),
                    f64::from(xf).abs() * ERF_BOUND,
                )
            })
            .count();
        let erf_ok = outside == 0 && tells > 0;
        println!(
            "rule merger.gelu_erf   {outside} of {} outside the erf bound |x|·2⁻²¹, {differ} differ from the host rounding; premise: the tanh rule falls outside the bound at {tells} of the same values  {}",
            pre.len(),
            verdict(erf_ok)
        );
        ok &= erf_ok;
        let down_w = hf.bf16(names::mm_weight("down"))?;
        ok &= gemm_check(
            "merger.down",
            h,
            &down_w,
            Some(&last.merger.down.b),
            n_tok,
            hp.out_dim,
            merged,
            &tapped,
        );
        let vit_f = to_f32(vit);
        let fm = by_windows(&vit_f, dim, |c| final_merger_f32(&last.merger, c, hp.eps));
        let vit_nudged = to_f32(&nudged(vit));
        let fm_nudged = by_windows(&vit_nudged, dim, |c| {
            final_merger_f32(&last.merger, c, hp.eps)
        });
        ok &= teacher_forced("final_merger", &tapped, &fm, &fm_nudged, hp.out_dim);

        // (d) three images
        let mut first = Vec::new();
        for (name, g, rows) in [
            (GATE_IMAGE, &gate, vec![TOKENS_448]),
            ("crop-200x200", &small, vec![TOKENS_448]),
            (SLICING_IMAGE, &sliced, vec![TOKENS_SLICED; 5]),
        ] {
            let out = encode_tiles(&mut enc, stream, &g.tiles)?;
            let grids: Vec<usize> = g
                .tiles
                .iter()
                .map(|t| {
                    let (w, h) = t.plan.span_grid();
                    w * h
                })
                .collect();
            let counts: Vec<usize> = out.iter().map(|(r, _)| r.len() / hp.out_dim).collect();
            let pass = counts == rows
                && counts == grids
                && out
                    .iter()
                    .all(|(r, l)| finite(r) && r.iter().any(|&v| v != 0) && *l == LAUNCHES);
            println!(
                "image {name}: {} tiles, rows per tile {counts:?} (want {rows:?}, the grids give {grids:?}), all finite, {LAUNCHES} launches a tile: {}",
                out.len(),
                verdict(pass)
            );
            ok &= pass;
            first.push((name, g, out));
        }
        // The gate image and the small one again, after the largest in the reused buffers: the
        // gate image's two plain encodes equal each other and the tapped run, and both are
        // bit-identical to their first encodes.
        let again_gate = encode_tiles(&mut enc, stream, &gate.tiles)?;
        let again_small = encode_tiles(&mut enc, stream, &small.tiles)?;
        ok &= eager_twice(
            GATE_IMAGE,
            (&tapped, tapped_launches),
            &(first[0].2[0].0.clone(), first[0].2[0].1),
            &again_gate[0].0,
            LAUNCHES,
        );
        let pass = again_gate == first[0].2 && again_small == first[1].2;
        println!(
            "reuse {GATE_IMAGE} and crop-200x200 after {SLICING_IMAGE} ({SLICED_PATCHES} patches a tile): bit-identical to their first encodes: {}",
            verdict(pass)
        );
        ok &= pass;

        // (e) mainline's nodes
        ok &= mainline_clauses(&mut enc, stream, &hp)?;
        Ok(ok)
    }
}
