//! GPU gate for NVFP4's dense gemv (`bloomery_gpu::mxfp4`: `Nvfp4Kernels`,
//! the FP4 owner's rule under `Nvfp4`) at Nemotron 3.5 Bolt's two Mamba-2
//! projections, `ssm_in` (18,560 rows × 4,096) and `ssm_out` (4,096 × 8,192),
//! on synthetic tensors written to a GGUF and loaded through the engine's
//! own path: `Weights::load_where` (the `CardFormat::Nvfp4Planes` upload,
//! the repack and the `.scale` read), `site::file_site`, `site::site` and
//! `site::kgemv` over the q8_1 rows. Its own gate, not `gate_dspark_experts`'
//! (MXFP4's): that one runs routed stacks of the DSpark draft's real file;
//! this one a dense site with a per-tensor scale on no model file.
//!
//! The tensors: every scale byte drawn from the 127 UE4M3 bytes the format
//! defines (0x00..=0x7E, the subnormals among them), every code byte from all
//! 256, and a per-tensor scale that is no power of two. Activations uniform
//! on ±2. For each shape at m = 1 and at m > 1 columns (3 for `ssm_in`, 8 for
//! `ssm_out`), every output is checked:
//! - `written`: finite, over a NaN sentinel the launch must overwrite;
//! - `band`: within the derived band of the reference `scale · Σ_j w_j ·
//!   q_j d_j` in f64, `w` the host dequant (`gguf::quant::dequant_row`, R0's
//!   port of mainline's `dequantize_row_nvfp4`) and `q_j d_j` the q8_1
//!   activation the host transcription of the quantizer gives
//!   (`act_rule::q8_1_codes`; the card's block scales are checked equal to
//!   its bits first);
//! - `bits`: equal bit for bit to the host transcription of the rule
//!   (`lane_partial_host::<Nvfp4, 2>` per lane, the warp butterfly, the multiply
//!   by the scale).
//!
//! A weight whose `.scale` sits in another shard of a split loads with that scale
//! (`two_shards`).
//!
//! `nvfp4_site_k64_through_sel32`: a site whose width is a multiple of 64 and no
//! multiple of 256, Lightning's shared-expert up (3,712 rows of 2,688) and down (2,688
//! rows of 3,712), reads the 32-value rows (`site::reads_at`) and runs
//! `site::nvfp4_sel32_gemv`: `written`, and `bits` equal to the host rule
//! (`lane_partial_host::<Nvfp4, 0>`, the butterfly, the multiply by the scale) with every
//! output inside the same band as above (a 32-value block a scale), no fault raised. A site
//! that read the q8_1 rows at these widths is refused by its launch contract.
//!
//! The routed pair (`bloomery_gpu::mxfp4_sel`: `Nvfp4SelKernels`) at
//! Qwen3.8's widths, the gate and up stacks 640 rows of 2,560 an expert and
//! the down 2,560 rows of 640, six experts in the file, each with a scale of
//! its own in the stack's `.scale` `[6]`. The stacks load through
//! `Weights::load_rows` on a plan row that puts experts 1, 3 and 4 on the
//! card and the rest on another card, so the planes and the scales hold
//! those three in that order:
//! - `routed_load`: the resident scales are the file's of experts 1, 3, 4;
//! - `routed_gate_up` over a `Q8Act` of two tokens, three slots each (one a
//!   `HOST` slot): `bits`, `h` equal to the host rule's
//!   (`lane_partial_host::<Nvfp4, 2>`, the butterfly, the expert's scale, then
//!   `swiglu_clamp`) bit for bit; `band`, each host gate and up dot within
//!   the band above of `scale · Σ w_j q_j d_j` in f64 (`dequant_row`); the
//!   `HOST` slot's rows at the sentinel;
//! - `routed_down` over the 32-value form (`Q8Blocks32`, its codes read
//!   back) of six slots: the same `bits` (`lane_partial_host::<Nvfp4, 0>`)
//!   and `band` (one `d` a 32-value block), the `HOST` slot at the sentinel;
//! - `routed_ids`: an id past the card's three experts raises `ExpertId` in
//!   each launch; the gate·up writes its rows NaN and the next clean launch
//!   raises nothing, the down leaves its rows at the sentinel and writes the
//!   other slots';
//! - `routed_overflow`: a gate·up whose up sum overflows to NaN on finite
//!   input raises `F32Product` and writes its rows NaN;
//! - `routed_f32_finite_scaled_word`: the expert select over f32 rows
//!   (`enqueue_nvfp4_gemv_sel_f32`, `site::resident_sel_f32`'s NVFP4 arm) of
//!   one row of 64 values, code 7
//!   under scale byte 0x01, against activations of 2^125: the weight times
//!   the activation is finite, so the row is the f64 reference's f32 bits and
//!   no fault is raised (a code times the activation before the scale is
//!   past f32's range).
//! - `routed_f32_alignment`: aligned and four-byte-offset activation rows
//!   use the same word reduction: cancellation yields zero, and overflowing
//!   lane sums raise `F32Product` and write NaN for either address.
//!
//! And the load's refusals, each by name: a scale that is NaN, a tensor with
//! no `.scale` beside it, and a scale byte UE4M3 does not define (0x7F).
//!
//! The band. Per sub-block of 16 values the card forms `d_s · s_s` exactly
//! (`d_s` a 4-bit significand, `|s_s| ≤ 16 · 12 · 127` an integer) — which is
//! `Σ_j w_j q_j` over the sub-block, `w_j = d_s · code_j` exact — and adds
//! `(d_s · s_s) · d_x` into its lane by one fused multiply-add: `n = 2 ·
//! n_units / 32` roundings a lane, five butterfly additions, one multiply by
//! the scale. So `|ours − ref| ≤ γ(n + 6) · |scale| · Σ_s |d_s s_s d_x|`, the
//! sum over the row's sub-blocks, plus the f64 reference's own `n_sub · u64`
//! of the same sum. The bound is the worst case, every error aligned.
//!
//! `--bench-kernels` runs no case: it prices the gemv's launch at both shapes
//! (`bench op=nvfp4_18560`, `bench op=nvfp4_4096`, one column) and the routed
//! pair at Qwen3.8's widths, one token of eight distinct experts a launch of
//! a stack of 128, each launch of a burst the next eight: a launch's experts
//! were last read 15 launches before, with 110 MB or more of other experts
//! read since, more than a 96 MB L2 holds
//! (`bench op=nvfp4_sel_gate_up`, `bench op=nvfp4_sel_down`), through
//! `shared/bench.rs`'s burst and replay, timed by `tools/ref/time-gate.sh`
//! under the lease; `--bench-arm <op>` keeps the arms whose name starts with
//! `<op>` (`nvfp4` is all four). Nothing there asserts on a result.

#[path = "shared/bench.rs"]
mod bench_arm;

fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_nvfp4", gate::run())
}

mod gate {
    use bloomery_gpu::mxfp4::{NV_BLOCK_BYTES, NV_BLOCK_VALUES, Nvfp4, lane_partial_host};
    use bloomery_gpu::q5::Q8Blocks32;
    use bloomery_gpu::site::{self, Form, KGemvKernels, SiteTy};
    use bloomery_gpu::weights::{DevWeight, Weights};
    use bloomery_gpu::{Gpu, Q8Act};
    use bloomery_gpu_gates::act_rule::{Q8_1_BLOCK, q8_1_codes};
    use bloomery_gpu_gates::rounding::{U64, butterfly, gamma};
    use bloomery_gpu_gates::{GateError, GateRng, checks_failed, par_map, verdict, write_gguf};
    use cuda_core::DeviceBuffer;
    use gguf::quant::{GgmlType, dequant_row};
    use gguf::{Split, Value};
    use std::path::Path;

    /// A Bolt Mamba-2 projection: its tensor stem, rows, row width, the
    /// per-tensor scale written beside it and the column counts it runs at.
    pub(crate) struct Shape {
        pub(crate) stem: &'static str,
        pub(crate) rows: usize,
        pub(crate) k: usize,
        pub(crate) scale: f32,
        pub(crate) ms: [usize; 2],
        pub(crate) seed: u64,
    }

    /// `ssm_in` and `ssm_out` of a Bolt Mamba-2 layer (`/workspace/bolt-arch.md`
    /// §4: in_proj 18,560 × 4,096, out_proj 4,096 × 8,192).
    pub(crate) const SHAPES: [Shape; 2] = [
        Shape {
            stem: "blk.0.ssm_in",
            rows: 18_560,
            k: 4096,
            scale: 0.013_71,
            ms: [1, 3],
            seed: 0x5eed_0001,
        },
        Shape {
            stem: "blk.0.ssm_out",
            rows: 4096,
            k: 8192,
            scale: 1.732_9,
            ms: [1, 8],
            seed: 0x5eed_0002,
        },
    ];

    /// Lightning's shared-expert up and down: widths of whole 64-value blocks that no q8_1
    /// super-block of 256 values divides, 2,688 and 3,712 (`nvfp4_site_k64_through_sel32`).
    pub(crate) const SEL32_SHAPES: [Shape; 2] = [
        Shape {
            stem: "blk.0.ffn_up_shexp",
            rows: 3712,
            k: 2688,
            scale: 0.037_13,
            ms: [1, 3],
            seed: 0x5eed_0201,
        },
        Shape {
            stem: "blk.0.ffn_down_shexp",
            rows: 2688,
            k: 3712,
            scale: 0.413_7,
            ms: [1, 8],
            seed: 0x5eed_0202,
        },
    ];

    /// `rows` NVFP4 rows of `k` values drawn from `seed`: each block's four
    /// scale bytes uniform over the 127 defined UE4M3 bytes, its 32 code bytes
    /// uniform over all 256.
    pub(crate) fn nvfp4_bytes(rows: usize, k: usize, seed: u64) -> Vec<u8> {
        let mut rng = GateRng(seed);
        let blocks = rows * k / NV_BLOCK_VALUES;
        let mut out = Vec::with_capacity(blocks * NV_BLOCK_BYTES);
        for _ in 0..blocks {
            for _ in 0..4 {
                out.push(rng.below(0x7F) as u8);
            }
            for _ in 0..32 {
                out.push(rng.below(256) as u8);
            }
        }
        out
    }

    /// One tensor of a synthetic file: name, dims, type, bytes.
    type Decl = (String, Vec<u64>, GgmlType, Vec<u8>);

    /// An NVFP4 weight of `dims` and its scales (one, or one an expert of a
    /// stack) as two tensors of a file.
    fn weight_decls(stem: &str, dims: &[usize], bytes: Vec<u8>, scales: &[f32]) -> [Decl; 2] {
        [
            (
                format!("{stem}.weight"),
                dims.iter().map(|&d| d as u64).collect(),
                GgmlType::NVFP4,
                bytes,
            ),
            (
                format!("{stem}.scale"),
                vec![scales.len() as u64],
                GgmlType::F32,
                scales.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
        ]
    }

    /// Rows `rows` of `k` values of the NVFP4 `bytes`, each dequantized by
    /// the host's port: the reference's weights.
    fn dequant_rows(bytes: &[u8], rows: std::ops::Range<usize>, k: usize) -> Vec<Vec<f32>> {
        let row_bytes = k / NV_BLOCK_VALUES * NV_BLOCK_BYTES;
        par_map(rows.len(), |i| {
            let r = rows.start + i;
            let mut v = vec![0.0f32; k];
            dequant_row(
                GgmlType::NVFP4,
                &bytes[r * row_bytes..][..row_bytes],
                &mut v,
            )
            .unwrap_or_else(|e| panic!("row {r}: {e}"));
            v
        })
    }

    /// The host rule of one row against one column (codes `q`, a scale `d`
    /// each `block` values: 128 for the q8_1 columns, 32 for the 32-value
    /// form), times the tensor's `scale` (module doc): its value, the f64
    /// reference `scale · Σ w_j q_j d` over the row's dequant `w`, and the
    /// derived band about it.
    fn host_rule(
        row: &[u8],
        w: &[f32],
        (q, d): (&[i8], &[f32]),
        block: usize,
        scale: f32,
    ) -> (f32, f64, f64) {
        let lanes: [f32; 32] = std::array::from_fn(|lane| {
            if block == 32 {
                lane_partial_host::<Nvfp4, 0>(row, q, d, lane)
            } else {
                lane_partial_host::<Nvfp4, 2>(row, q, d, lane)
            }
        });
        let host = butterfly(lanes) * scale;
        let (mut sum, mut mag) = (0.0f64, 0.0f64);
        for (sub, wv) in w.as_chunks::<16>().0.iter().enumerate() {
            let t: f64 = wv
                .iter()
                .zip(&q[16 * sub..])
                .map(|(&wj, &qj)| f64::from(wj) * f64::from(qj))
                .sum::<f64>()
                * f64::from(d[16 * sub / block]);
            sum += t;
            mag += t.abs();
        }
        let k = w.len();
        let s = f64::from(scale);
        let band = (gamma(2 * (k / 32).div_ceil(32) + 6) + (k / 16) as f64 * U64) * s.abs() * mag;
        (host, s * sum, band)
    }

    /// One output against [`host_rule`]'s `(host, reference, band)`: whether
    /// its bits are the host's, and its distance from the reference with the
    /// band.
    fn judge(got: f32, (host, reference, band): (f32, f64, f64)) -> (bool, f64, f64) {
        (
            got.to_bits() == host.to_bits(),
            (f64::from(got) - reference).abs(),
            band,
        )
    }

    /// The `bits` and `band` clauses over each output's [`judge`]: how many
    /// match the host's bits, whether all lie in their band, and the worst
    /// distance over the band.
    fn tally(per: &[(bool, f64, f64)]) -> (usize, bool, f64) {
        let bits = per.iter().filter(|p| p.0).count();
        let band = per.iter().all(|p| p.1 <= p.2);
        let worst = per
            .iter()
            .map(|p| if p.2 > 0.0 { p.1 / p.2 } else { p.1 })
            .fold(0.0f64, f64::max);
        (bits, band, worst)
    }

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let bench = std::env::args().any(|a| a == "--bench-kernels");
        let only = if bench {
            bloomery_gpu_gates::bench_arm(
                std::env::args(),
                SHAPES
                    .iter()
                    .map(|s| format!("nvfp4_{}", s.rows))
                    .chain(bench::ROUTED_ARMS.into_iter().map(str::to_owned)),
            )?
        } else {
            None
        };
        let gpu = Gpu::new()?;
        if bench {
            return bench::run(&gpu, only.as_deref());
        }
        println!("nvfp4 device={}", gpu.device_name()?);
        let dir = std::env::temp_dir().join(format!("bloomery-nvfp4-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let outcome = cases(&gpu, &dir);
        let cleanup = std::fs::remove_dir_all(&dir).map_err(|e| format!("cleanup: {e}"));
        let ok = outcome?;
        cleanup?;
        if ok { Ok(()) } else { Err(checks_failed()) }
    }

    fn cases(gpu: &Gpu, dir: &Path) -> Result<bool, GateError> {
        let path = dir.join("bolt-ssm.gguf");
        let mut decls = Vec::new();
        let mut raw = Vec::new();
        for s in &SHAPES {
            let bytes = nvfp4_bytes(s.rows, s.k, s.seed);
            raw.push(bytes.clone());
            decls.extend(weight_decls(s.stem, &[s.k, s.rows], bytes, &[s.scale]));
        }
        let mut raw32 = Vec::new();
        for s in &SEL32_SHAPES {
            let bytes = nvfp4_bytes(s.rows, s.k, s.seed);
            raw32.push(bytes.clone());
            decls.extend(weight_decls(s.stem, &[s.k, s.rows], bytes, &[s.scale]));
        }
        // The refusals' tensors: 8 rows of 256, scale 1 unless named.
        let small = |seed| nvfp4_bytes(8, 256, seed);
        decls.extend(weight_decls("nan", &[256, 8], small(7), &[f32::NAN]));
        decls.push(weight_decls("lone", &[256, 8], small(8), &[1.0])[0].clone());
        let mut undefined = small(9);
        undefined[5 * NV_BLOCK_BYTES + 1] = 0x7F;
        decls.extend(weight_decls("undef", &[256, 8], undefined, &[1.0]));
        for i in 0..6 {
            let mut scales = [1.0; 6];
            scales[i] = f32::NAN;
            decls.extend(weight_decls(
                &format!("expert_nan_{i}"),
                &[256, 1, 6],
                nvfp4_bytes(6, 256, 10),
                &scales,
            ));
        }
        for count in [5, 7] {
            decls.extend(weight_decls(
                &format!("expert_count_{count}"),
                &[256, 1, 6],
                nvfp4_bytes(6, 256, 10),
                &vec![1.0; count],
            ));
        }
        write_gguf(&path, &[], &decls)?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let stream = gpu.stream();
        let kk = KGemvKernels::load(gpu)?;
        let mut ok = true;
        for (s, bytes) in SHAPES.iter().zip(&raw) {
            ok &= shape_case(gpu, &kk, &split, s, bytes)?;
        }
        for (s, bytes) in SEL32_SHAPES.iter().zip(&raw32) {
            ok &= sel32_case(gpu, &kk, &split, s, bytes)?;
        }
        for (stem, want) in [
            ("nan", "not a finite value"),
            ("lone", "lone.scale is not in the file"),
            (
                "undef",
                "row 1 block 1 sub-block 1: scale byte 0x7f is no UE4M3 scale",
            ),
        ]
        .map(|(stem, want)| (stem.to_string(), want.to_string()))
        .into_iter()
        .chain((0..6).map(|i| {
            (
                format!("expert_nan_{i}"),
                format!("[{i}] is NaN, not a finite value"),
            )
        }))
        .chain([5, 7].map(|n| {
            (
                format!("expert_count_{n}"),
                "one f32 value an expert".into(),
            )
        })) {
            let name = format!("{stem}.weight");
            let got = match Weights::load_where(stream, &split, |n| n == name) {
                Ok(_) => "loaded".to_string(),
                Err(e) => e.to_string(),
            };
            let pass = got.contains(&want);
            println!(
                "nvfp4 clause=refusal tensor={name} want={want:?} got={got:?} {}",
                verdict(pass)
            );
            ok &= pass;
        }
        ok &= two_shards(gpu, dir)?;
        ok &= routed::run(gpu, dir)?;
        Ok(ok)
    }

    /// A weight in shard 0 of a two-shard split whose `.scale` is in shard 1 loads with that scale:
    /// the scale is found across the split, not in the weight's own shard.
    fn two_shards(gpu: &Gpu, dir: &Path) -> Result<bool, GateError> {
        const SCALE: f32 = 0.625;
        let [weight, scale] = weight_decls("x2", &[256, 8], nvfp4_bytes(8, 256, 11), &[SCALE]);
        let keys = |no: u16| {
            vec![
                ("split.no".to_string(), Value::U16(no)),
                ("split.count".to_string(), Value::U16(2)),
                ("split.tensors.count".to_string(), Value::I32(2)),
            ]
        };
        let first = dir.join("x2-00001-of-00002.gguf");
        write_gguf(&first, &keys(0), &[weight])?;
        write_gguf(&dir.join("x2-00002-of-00002.gguf"), &keys(1), &[scale])?;
        let split = Split::open(&first).map_err(|e| format!("open {}: {e}", first.display()))?;
        let got = match Weights::load_where(gpu.stream(), &split, |n| n == "x2.weight") {
            Ok(w) => match w.get("x2.weight") {
                Some(DevWeight::Nvfp4 { scale, .. }) => format!("scale {scale}"),
                other => format!("not an NVFP4 weight: {}", other.is_some()),
            },
            Err(e) => e.to_string(),
        };
        let want = format!("scale {SCALE}");
        let pass = got == want;
        println!(
            "nvfp4 clause=two_shards tensor=x2.weight want={want:?} got={got:?} {}",
            verdict(pass)
        );
        Ok(pass)
    }

    /// One shape at each of its column counts: the clauses of the module doc.
    fn shape_case(
        gpu: &Gpu,
        kk: &KGemvKernels,
        split: &Split,
        s: &Shape,
        bytes: &[u8],
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let name = format!("{}.weight", s.stem);
        let (rows, k) = (s.rows, s.k);
        let ty = site::file_site(split, "gate_nvfp4", &name, (rows, k), &[SiteTy::Nvfp4])?;
        let w = Weights::load_where(stream, split, |n| n == name)?;
        site::site(&w, "gate_nvfp4", &name, (rows, k), ty)?;
        let row_bytes = k / NV_BLOCK_VALUES * NV_BLOCK_BYTES;
        let wf = dequant_rows(bytes, 0..rows, k);
        let mut ok = true;
        for (i, &m) in s.ms.iter().enumerate() {
            let mut rng = GateRng(s.seed ^ (0x1000 + i as u64));
            let x: Vec<f32> = (0..m * k).map(|_| rng.uni(-2.0, 2.0)).collect();
            let x_dev = DeviceBuffer::from_host(stream, &x)?;
            let mut act = Q8Act::with_k(stream, m, k)?;
            gpu.enqueue_quantize_q8_1_layer(&x_dev, &mut act, 0)?;
            let mut y_dev = DeviceBuffer::from_host(stream, &vec![f32::NAN; m * rows])?;
            site::kgemv(
                gpu,
                kk,
                (ty, &w, &name),
                &act,
                gpu.unlabelled_sink(),
                &mut y_dev,
            )?;
            stream.synchronize()?;
            let y = y_dev.to_host_vec(stream)?;
            let d8_card = act.d8().to_host_vec(stream)?;
            let cols: Vec<(Vec<i8>, Vec<f32>)> =
                (0..m).map(|c| q8_1_codes(&x[c * k..][..k])).collect();
            let d8_same = cols.iter().enumerate().all(|(c, (_, d))| {
                d.iter()
                    .zip(&d8_card[c * 2 * (k / 256)..])
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            });
            let written = y.iter().all(|v| v.is_finite());
            // Per (column, row): the host rule's bits, the reference, the band.
            let per: Vec<(bool, f64, f64)> = par_map(m * rows, |at| {
                let (c, r) = (at / rows, at % rows);
                let (q, d) = &cols[c];
                let row = &bytes[r * row_bytes..][..row_bytes];
                judge(y[at], host_rule(row, &wf[r], (q, d), Q8_1_BLOCK, s.scale))
            });
            let (bits_n, band_ok, worst) = tally(&per);
            let bits = bits_n == per.len();
            let tag = format!("shape={} rows={rows} k={k} m={m}", s.stem);
            println!("nvfp4 clause=written {tag} {}", verdict(written));
            println!(
                "nvfp4 clause=d8 {tag} card_scales_equal_host={d8_same} {}",
                verdict(d8_same)
            );
            println!(
                "nvfp4 clause=band {tag} worst_err_over_band={worst:.3e} {}",
                verdict(band_ok)
            );
            println!(
                "nvfp4 clause=bits {tag} equal={bits_n}/{} {}",
                m * rows,
                verdict(bits)
            );
            ok &= written && d8_same && band_ok && bits;
        }
        Ok(ok)
    }

    /// One shape of [`SEL32_SHAPES`] at each of its column counts: the `nvfp4_site_k64_through_sel32`
    /// clause (module doc). The form the width reads picks the launch, as a body's site does
    /// ([`site::reads_at`]); a width that read the q8_1 rows would launch `site::kgemv` and end on
    /// its contract's refusal.
    fn sel32_case(
        gpu: &Gpu,
        kk: &KGemvKernels,
        split: &Split,
        s: &Shape,
        bytes: &[u8],
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let name = format!("{}.weight", s.stem);
        let (rows, k) = (s.rows, s.k);
        let ty = site::file_site(split, "gate_nvfp4", &name, (rows, k), &[SiteTy::Nvfp4])?;
        let w = Weights::load_where(stream, split, |n| n == name)?;
        site::site(&w, "gate_nvfp4", &name, (rows, k), ty)?;
        let row_bytes = k / NV_BLOCK_VALUES * NV_BLOCK_BYTES;
        let wf = dequant_rows(bytes, 0..rows, k);
        let form = site::reads_at(ty, k);
        let mut ok = true;
        for (i, &m) in s.ms.iter().enumerate() {
            let mut rng = GateRng(s.seed ^ (0x1000 + i as u64));
            let x: Vec<f32> = (0..m * k).map(|_| rng.uni(-2.0, 2.0)).collect();
            let x_dev = DeviceBuffer::from_host(stream, &x)?;
            let mut y_dev = DeviceBuffer::from_host(stream, &vec![f32::NAN; m * rows])?;
            let tag = format!("shape={} rows={rows} k={k} m={m}", s.stem);
            if form != Form::Q8Blocks32 {
                let mut act = Q8Act::with_k(stream, m, k)?;
                gpu.enqueue_quantize_q8_1_layer(&x_dev, &mut act, 0)?;
                site::kgemv(
                    gpu,
                    kk,
                    (ty, &w, &name),
                    &act,
                    gpu.unlabelled_sink(),
                    &mut y_dev,
                )?;
                println!(
                    "nvfp4 clause=nvfp4_site_k64_through_sel32 {tag} form={form:?} \
                     want=Q8Blocks32 {}",
                    verdict(false)
                );
                ok = false;
                continue;
            }
            let mut act = Q8Blocks32::new(stream, k, m)?;
            gpu.q5()
                .enqueue_quantize_q8(stream, &x_dev, &mut act, gpu.unlabelled_sink())?;
            site::nvfp4_sel32_gemv(
                gpu,
                kk,
                (ty, &w, &name),
                &act,
                gpu.unlabelled_sink(),
                &mut y_dev,
            )?;
            stream.synchronize()?;
            let y = y_dev.to_host_vec(stream)?;
            let fault = gpu.take_fault()?;
            let back = act.readback(stream)?;
            let cols: Vec<(Vec<i8>, Vec<f32>)> = (0..m).map(|c| back.column(c, k, m)).collect();
            let written = y.iter().all(|v| v.is_finite());
            let per: Vec<(bool, f64, f64)> = par_map(m * rows, |at| {
                let (c, r) = (at / rows, at % rows);
                let (q, d) = &cols[c];
                let row = &bytes[r * row_bytes..][..row_bytes];
                judge(y[at], host_rule(row, &wf[r], (q, d), 32, s.scale))
            });
            let (bits_n, band_ok, worst) = tally(&per);
            let pass = written && fault.is_none() && band_ok && bits_n == per.len();
            println!(
                "nvfp4 clause=nvfp4_site_k64_through_sel32 {tag} form={form:?} written={written} \
                 fault_none={} worst_err_over_band={worst:.3e} bits_equal={bits_n}/{} {}",
                fault.is_none(),
                m * rows,
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    /// Qwen3.8's hidden width and routed width: the routed clauses' and the
    /// routed bench arms' stacks.
    const ROUTED_N: usize = 2560;
    const ROUTED_FF: usize = 640;

    /// The routed pair's clauses (module doc).
    mod routed {
        use super::{
            ROUTED_FF, ROUTED_N, dequant_rows, host_rule, judge, nvfp4_bytes, tally, weight_decls,
            write_gguf,
        };
        use bloomery_gpu::hybrid::HOST;
        use bloomery_gpu::kquant::act::{Act, swiglu_clamp};
        use bloomery_gpu::mxfp4::{NV_BLOCK_BYTES, NV_BLOCK_VALUES, Nvfp4, Planes, repack};
        use bloomery_gpu::mxfp4_sel::{Nvfp4Down, Nvfp4GateUp, Nvfp4SelKernels, Nvfp4Stack};
        use bloomery_gpu::q5::Q8Blocks32;
        use bloomery_gpu::weights::Weights;
        use bloomery_gpu::{DeviceTensor, Gpu, GpuError, Q8Act, Window};
        use bloomery_gpu_gates::act_rule::{Q8_1_BLOCK, q8_1_codes};
        use bloomery_gpu_gates::{GateError, GateRng, par_map, verdict};
        use cuda_core::DeviceBuffer;
        use gguf::Split;
        use model::placement::{self, ExpertList, ModelTensor, ModelTensors, Role};
        use std::path::Path;

        /// Experts in the file.
        const EXPERTS: usize = 6;
        /// The experts the card holds, in its order; the rest are another card's.
        const CARD: [u32; 3] = [1, 3, 4];
        /// Slots a token.
        const SPC: usize = 3;
        /// Two tokens' places: every card expert, a duplicate and a host slot.
        const SEL: [u32; 6] = [0, 2, HOST, 1, 0, 2];
        /// The SwiGLU clamp, whose `expf` the host and the card round alike.
        const LIMIT: f32 = 7.0;
        /// What a slot the launch leaves alone still holds.
        const SENT: f32 = 1.0e30;

        /// A stack's part, rows and row width an expert, and seed.
        const PARTS: [(&str, usize, usize, u64); 3] = [
            ("gate", ROUTED_FF, ROUTED_N, 0x5eed_0101),
            ("up", ROUTED_FF, ROUTED_N, 0x5eed_0102),
            ("down", ROUTED_N, ROUTED_FF, 0x5eed_0103),
        ];

        /// A stack as the file holds it, and each card expert's rows
        /// dequantized, by card place.
        struct Stack {
            bytes: Vec<u8>,
            rows: usize,
            k: usize,
            scales: Vec<f32>,
            w: Vec<Vec<Vec<f32>>>,
        }

        impl Stack {
            /// Row `r` of the expert at card place `p` against one column,
            /// by [`host_rule`] with the expert's scale.
            fn rule(
                &self,
                p: u32,
                r: usize,
                col: (&[i8], &[f32]),
                block: usize,
            ) -> (f32, f64, f64) {
                let e = CARD[p as usize] as usize;
                let row_bytes = self.k / NV_BLOCK_VALUES * NV_BLOCK_BYTES;
                let row = &self.bytes[(e * self.rows + r) * row_bytes..][..row_bytes];
                host_rule(row, &self.w[p as usize][r], col, block, self.scales[e])
            }
        }

        pub(super) fn run(gpu: &Gpu, dir: &Path) -> Result<bool, GateError> {
            let stream = gpu.stream();
            let mut decls = Vec::new();
            let mut stacks = Vec::new();
            for (i, &(part, rows, k, seed)) in PARTS.iter().enumerate() {
                let bytes = nvfp4_bytes(EXPERTS * rows, k, seed);
                let mut rng = GateRng(seed ^ 0x5ca1e);
                let scales: Vec<f32> = (0..EXPERTS)
                    .map(|_| rng.uni(0.004, 0.02) * (1 + i) as f32)
                    .collect();
                let stem = format!("blk.0.ffn_{part}_exps");
                decls.extend(weight_decls(
                    &stem,
                    &[k, rows, EXPERTS],
                    bytes.clone(),
                    &scales,
                ));
                let w = CARD
                    .iter()
                    .map(|&e| dequant_rows(&bytes, e as usize * rows..(e as usize + 1) * rows, k))
                    .collect();
                stacks.push(Stack {
                    bytes,
                    rows,
                    k,
                    scales,
                    w,
                });
            }
            let path = dir.join("routed.gguf");
            write_gguf(&path, &[], &decls)?;
            let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
            let names: Vec<String> = PARTS
                .iter()
                .map(|(part, ..)| format!("blk.0.ffn_{part}_exps.weight"))
                .collect();
            let model = ModelTensors {
                tensors: names
                    .iter()
                    .map(|name| {
                        let (_, t) = split.find(name).ok_or(format!("{name} not in the file"))?;
                        Ok(ModelTensor {
                            name: name.clone(),
                            shard: 0,
                            layer: Some(0),
                            role: Role::RoutedExperts,
                            ty: t.ty,
                            dims: t.dims.clone(),
                            file_bytes: t.nbytes,
                            gathered_rows: None,
                        })
                    })
                    .collect::<Result<_, GateError>>()?,
                layers: 1,
                experts: EXPERTS as u64,
                experts_used: 2,
                bf16_words: |_| false,
            };
            let list = |ids: &[u32]| ExpertList::new(ids.to_vec(), EXPERTS as u64);
            let rows = model
                .tensors
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    placement::routed_row_on(
                        i,
                        t,
                        0,
                        vec![(0, list(&CARD)?), (1, list(&[0, 2, 5])?)],
                        &model,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("routed rows: {e}"))?;
            let w = Weights::load_rows(stream, &split, &model, &rows, 0)?;
            let planes: Vec<_> = names
                .iter()
                .map(|n| w.nvfp4_stack("gate_nvfp4", n))
                .collect::<Result<_, GpuError>>()?;
            let mut ok = true;
            for (n, (st, pl)) in names.iter().zip(stacks.iter().zip(&planes)) {
                let got = pl.scales.buf().to_host_vec(stream)?;
                let want: Vec<f32> = CARD.iter().map(|&e| st.scales[e as usize]).collect();
                let pass = got.len() == want.len()
                    && got
                        .iter()
                        .zip(&want)
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                println!(
                    "nvfp4 clause=routed_load tensor={n} scales={got:?} {}",
                    verdict(pass)
                );
                ok &= pass;
            }
            let nv = Nvfp4SelKernels::load(gpu.context(), gpu.fault_word())?;
            let sel = DeviceBuffer::from_host(stream, &SEL)?;
            let rule = Act::SwigluClamp { limit: LIMIT };

            // The gate·up over two tokens.
            let m = SEL.len() / SPC;
            let mut rng = GateRng(0x5eed_0201);
            let x: Vec<f32> = (0..m * ROUTED_N).map(|_| rng.uni(-2.0, 2.0)).collect();
            let x_dev = DeviceBuffer::from_host(stream, &x)?;
            let mut act = Q8Act::with_k(stream, m, ROUTED_N)?;
            gpu.enqueue_quantize_q8_1_layer(&x_dev, &mut act, 0)?;
            let gate_up = |sel: &DeviceBuffer<u32>, n_slots: usize| {
                let a = Nvfp4GateUp {
                    gate: planes[0],
                    up: planes[1],
                    act: &act,
                    sel,
                    n_slots,
                    rows_per_expert: ROUTED_FF,
                    slots_per_col: SPC,
                    rule,
                };
                let mut h = DeviceBuffer::from_host(stream, &vec![SENT; n_slots * ROUTED_FF])?;
                nv.enqueue_gate_up(stream, &a, gpu.unlabelled_sink(), &mut h)?;
                Ok::<_, GateError>((h.to_host_vec(stream)?, gpu.take_fault()?))
            };
            let (h, fault) = gate_up(&sel, SEL.len())?;
            let cols: Vec<(Vec<i8>, Vec<f32>)> = (0..m)
                .map(|c| q8_1_codes(&x[c * ROUTED_N..][..ROUTED_N]))
                .collect();
            let d8_card = act.d8().to_host_vec(stream)?;
            let d8_same = cols.iter().enumerate().all(|(c, (_, d))| {
                d.iter()
                    .zip(&d8_card[c * 2 * (ROUTED_N / 256)..])
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            });
            // The bits of the rule over the gate's and the up's host values,
            // and of those two sums the one farther into its band.
            let per: Vec<(bool, f64, f64)> = par_map(SEL.len() * ROUTED_FF, |at| {
                let (s, r) = (at / ROUTED_FF, at % ROUTED_FF);
                if SEL[s] == HOST {
                    return (true, 0.0, 0.0);
                }
                let (q, d) = &cols[s / SPC];
                let g = stacks[0].rule(SEL[s], r, (q, d), Q8_1_BLOCK);
                let u = stacks[1].rule(SEL[s], r, (q, d), Q8_1_BLOCK);
                let want = swiglu_clamp(g.0, u.0, LIMIT);
                let (_, ge, gb) = judge(g.0, g);
                let (_, ue, ub) = judge(u.0, u);
                let (e, b) = if ge * ub >= ue * gb {
                    (ge, gb)
                } else {
                    (ue, ub)
                };
                (h[at].to_bits() == want.to_bits(), e, b)
            });
            ok &= report(
                "routed_gate_up",
                d8_same && fault.is_none(),
                &per,
                &h,
                ROUTED_FF,
            );

            // The down over six slots of the 32-value form.
            let n_slots = SEL.len();
            let hx: Vec<f32> = (0..n_slots * ROUTED_FF)
                .map(|_| rng.uni(-3.0, 3.0))
                .collect();
            let hx_dev = DeviceBuffer::from_host(stream, &hx)?;
            let selected = [0u32, 1, 2, 0, 2, 1];
            let selected_dev = DeviceBuffer::from_host(stream, &selected)?;
            for (part, st) in stacks.iter().enumerate() {
                let (input, device, spc) = if part == 2 {
                    (&hx, &hx_dev, 1)
                } else {
                    (&x, &x_dev, SPC)
                };
                let args = bloomery_gpu::q8f32::SelArgs {
                    x: device,
                    sel: &selected_dev,
                    n_slots: selected.len(),
                    rows_per_expert: st.rows,
                    slots_per_col: spc,
                };
                let mut output = DeviceBuffer::zeroed(stream, selected.len() * st.rows)?;
                bloomery_gpu::site::resident_sel_f32(
                    gpu,
                    &w,
                    &names[part],
                    &args,
                    gpu.unlabelled_sink(),
                    &mut output,
                )?;
                let got = output.to_host_vec(stream)?;
                let mut worst = 0.0f64;
                let band = f64::from(f32::EPSILON) * (st.k.div_ceil(32) + 7) as f64 * 2.0;
                for (slot, &id) in selected.iter().enumerate() {
                    let scale = f64::from(st.scales[CARD[id as usize] as usize]);
                    for row in 0..st.rows {
                        let (mut sum, mut magnitude) = (0.0f64, 0.0f64);
                        for (&weight, &activation) in st.w[id as usize][row]
                            .iter()
                            .zip(&input[(slot / spc) * st.k..][..st.k])
                        {
                            let term = f64::from(weight) * f64::from(activation) * scale;
                            sum += term;
                            magnitude += term.abs();
                        }
                        worst = worst.max(
                            (f64::from(got[slot * st.rows + row]) - sum).abs()
                                / magnitude.max(f64::MIN_POSITIVE),
                        );
                    }
                }
                let pass = got.iter().all(|v| v.is_finite())
                    && worst <= band
                    && gpu.take_fault()?.is_none();
                println!(
                    "nvfp4 clause=routed_f32 part={part} rows={} k={} normalized_error={worst:.9} band={band:.9} {}",
                    st.rows,
                    st.k,
                    verdict(pass)
                );
                ok &= pass;
            }
            let mut nonfinite = hx[..ROUTED_FF].to_vec();
            nonfinite[0] = f32::NAN;
            let nonfinite_dev = DeviceBuffer::from_host(stream, &nonfinite)?;
            for (label, id, input, expected) in [
                ("id", CARD.len() as u32, &hx_dev, "expert_id"),
                ("nonfinite", 0, &nonfinite_dev, "f32_product"),
            ] {
                let ids = DeviceBuffer::from_host(stream, &[id])?;
                let args = bloomery_gpu::q8f32::SelArgs {
                    x: input,
                    sel: &ids,
                    n_slots: 1,
                    rows_per_expert: ROUTED_N,
                    slots_per_col: 1,
                };
                let mut output = DeviceBuffer::zeroed(stream, ROUTED_N)?;
                bloomery_gpu::site::resident_sel_f32(
                    gpu,
                    &w,
                    &names[2],
                    &args,
                    gpu.unlabelled_sink(),
                    &mut output,
                )?;
                let got = output.to_host_vec(stream)?;
                let named = gpu.take_fault()?.is_some_and(|f| {
                    GpuError::fault("routed_f32", f)
                        .to_string()
                        .contains(expected)
                });
                let pass = named && got.iter().all(|v| v.is_nan());
                println!(
                    "nvfp4 clause=routed_f32_refusal input={label} named={named} rows_nan={} {}",
                    got.iter().all(|v| v.is_nan()),
                    verdict(pass)
                );
                ok &= pass;
            }
            let mut act32 = Q8Blocks32::new(stream, ROUTED_FF, n_slots)?;
            gpu.q5()
                .enqueue_quantize_q8(stream, &hx_dev, &mut act32, gpu.unlabelled_sink())?;
            let back = act32.readback(stream)?;
            let cols32: Vec<_> = (0..n_slots)
                .map(|c| back.column(c, ROUTED_FF, n_slots))
                .collect();
            let mut y_dev = DeviceBuffer::from_host(stream, &vec![SENT; n_slots * ROUTED_N])?;
            let d = Nvfp4Down {
                w: planes[2],
                act: &act32,
                sel: &sel,
                n_slots,
                rows_per_expert: ROUTED_N,
                slots_per_col: SPC,
            };
            nv.enqueue_down(stream, &d, gpu.unlabelled_sink(), &mut y_dev)?;
            let y = y_dev.to_host_vec(stream)?;
            let fault = gpu.take_fault()?;
            let per: Vec<(bool, f64, f64)> = par_map(n_slots * ROUTED_N, |at| {
                let (s, r) = (at / ROUTED_N, at % ROUTED_N);
                if SEL[s] == HOST {
                    return (true, 0.0, 0.0);
                }
                let (q, d) = &cols32[s];
                judge(y[at], stacks[2].rule(SEL[s], r, (q, d), 32))
            });
            ok &= report("routed_down", fault.is_none(), &per, &y, ROUTED_N);

            // An id past the card's experts, one token.
            let bad = DeviceBuffer::from_host(stream, &[0u32, CARD.len() as u32, 1])?;
            let (h_bad, fault) = gate_up(&bad, SPC)?;
            let named = fault.is_some_and(|f| {
                GpuError::fault("gate_nvfp4", f)
                    .to_string()
                    .contains("expert_id")
            });
            let nan_rows = h_bad[ROUTED_FF..2 * ROUTED_FF].iter().all(|v| v.is_nan());
            let (_, next) = gate_up(&bad, 1)?;
            let pass = named && nan_rows && next.is_none();
            println!(
                "nvfp4 clause=routed_ids launch=gate_up fault_names_expert_id={named} \
                 rows_nan={nan_rows} next_clean={} {}",
                next.is_none(),
                verdict(pass)
            );
            // The same places through the down: the bad slot's rows stay at
            // the sentinel, the others are written.
            let mut y_bad = DeviceBuffer::from_host(stream, &vec![SENT; SPC * ROUTED_N])?;
            let d = Nvfp4Down {
                w: planes[2],
                act: &act32,
                sel: &bad,
                n_slots: SPC,
                rows_per_expert: ROUTED_N,
                slots_per_col: SPC,
            };
            nv.enqueue_down(stream, &d, gpu.unlabelled_sink(), &mut y_bad)?;
            let y_bad = y_bad.to_host_vec(stream)?;
            let fault = gpu.take_fault()?;
            let down_named = fault.is_some_and(|f| {
                GpuError::fault("gate_nvfp4", f)
                    .to_string()
                    .contains("expert_id")
            });
            let held = y_bad
                .chunks(ROUTED_N)
                .enumerate()
                .all(|(s, rows)| (s == 1) == rows.iter().all(|v| *v == SENT));
            let down_pass = down_named && held;
            println!(
                "nvfp4 clause=routed_ids launch=down fault_names_expert_id={down_named} \
                 bad_slot_at_sentinel_others_written={held} {}",
                verdict(down_pass)
            );
            let over = overflow(gpu, &nv)?;
            let scaled = finite_scaled_word(gpu)?;
            let aligned = f32_alignment(gpu)?;
            Ok(ok && pass && down_pass && over && scaled && aligned)
        }

        /// One expert-select launch of a one-row, one-expert NVFP4 stack over `x`:
        /// the row's f32 and the fault the launch raised.
        fn sel_f32(
            gpu: &Gpu,
            (qs, e, scales): (&DeviceTensor<u32>, &DeviceTensor<u32>, &DeviceTensor<f32>),
            x: &DeviceBuffer<f32>,
        ) -> Result<(f32, Option<bloomery_gpu::Fault>), GateError> {
            let stream = gpu.stream();
            let sel = DeviceBuffer::from_host(stream, &[0u32])?;
            let mut y = DeviceBuffer::from_host(stream, &[SENT])?;
            gpu.q8f32().enqueue_nvfp4_gemv_sel_f32(
                stream,
                Nvfp4Stack { qs, e, scales },
                &bloomery_gpu::q8f32::SelArgs {
                    x,
                    sel: &sel,
                    n_slots: 1,
                    rows_per_expert: 1,
                    slots_per_col: 1,
                },
                gpu.unlabelled_sink(),
                &mut y,
            )?;
            Ok((y.to_host_vec(stream)?[0], gpu.take_fault()?))
        }

        /// Unit weights against cancellation and overflowing lane sums,
        /// with identical activations at aligned and four-byte-offset bases.
        fn f32_alignment(gpu: &Gpu) -> Result<bool, GateError> {
            const K: usize = 64;
            let stream = gpu.stream();
            let row = [[0x38u8; 4].as_slice(), &[0x22u8; 32]].concat();
            let Planes { qs, e } = repack::<Nvfp4>(&row, 1, K)?;
            let qs = DeviceTensor::upload(stream, &qs, 1, K / 8)?;
            let e = DeviceTensor::upload(stream, &e, 1, K / NV_BLOCK_VALUES)?;
            let scales = DeviceTensor::upload(stream, &[1.0f32], 1, 1)?;
            let mut ok = true;
            for overflow in [false, true] {
                let mut x = vec![0.0f32; K];
                x[0] = if overflow { f32::MAX } else { 33554432.0 };
                x[1] = if overflow { f32::MAX } else { 1.0 };
                x[32] = -x[0];
                x[33] = if overflow { -f32::MAX } else { 0.0 };
                let mut padded = vec![0.0f32];
                padded.extend_from_slice(&x);
                let aligned = DeviceBuffer::from_host(stream, &x)?;
                let parent = DeviceBuffer::from_host(stream, &padded)?;
                let offset = Window::<f32>::of(&parent, size_of::<f32>(), K)?;
                for (alignment, x) in [("aligned", &aligned), ("offset4", &*offset)] {
                    let (got, fault) = sel_f32(gpu, (&qs, &e, &scales), x)?;
                    let named = fault.is_some_and(|f| {
                        GpuError::fault("gate_nvfp4", f)
                            .to_string()
                            .contains("f32_product")
                    });
                    let pass = if overflow {
                        named && got.is_nan()
                    } else {
                        fault.is_none() && got.to_bits() == 0.0f32.to_bits()
                    };
                    ok &= pass;
                    println!(
                        "nvfp4 clause=routed_f32_alignment alignment={alignment} overflow={overflow} \
                         got={got:e} fault_names_f32_product={named} {}",
                        verdict(pass)
                    );
                }
            }
            Ok(ok)
        }

        /// The `routed_f32_finite_scaled_word` clause (module doc): one expert
        /// of one row of 64 values, whose first code byte is 0x07 under scale
        /// byte 0x01 on every sub-block and the other codes 0, against 64
        /// activations of 2^125.
        fn finite_scaled_word(gpu: &Gpu) -> Result<bool, GateError> {
            const K: usize = 64;
            let stream = gpu.stream();
            let mut row = vec![0x01u8; 4];
            row.extend([0u8; 32]);
            row[4] = 0x07;
            let Planes { qs, e } = repack::<Nvfp4>(&row, 1, K)?;
            let (qs, e) = (
                DeviceTensor::upload(stream, &qs, 1, K / 8)?,
                DeviceTensor::upload(stream, &e, 1, K / NV_BLOCK_VALUES)?,
            );
            let scales = DeviceTensor::upload(stream, &[1.0f32], 1, 1)?;
            let x = vec![f32::from_bits((127 + 125) << 23); K];
            let want = dequant_rows(&row, 0..1, K)[0]
                .iter()
                .zip(&x)
                .map(|(&w, &a)| f64::from(w) * f64::from(a))
                .sum::<f64>() as f32;
            let x = DeviceBuffer::from_host(stream, &x)?;
            let (got, fault) = sel_f32(gpu, (&qs, &e, &scales), &x)?;
            let clean = fault.is_none();
            let pass = clean && want.is_finite() && got.to_bits() == want.to_bits();
            println!(
                "nvfp4 clause=routed_f32_finite_scaled_word got={got:e} want={want:e} \
                 no_fault={clean} {}",
                verdict(pass)
            );
            Ok(pass)
        }

        /// The `routed_overflow` clause (module doc): one expert of eight rows
        /// of 256 values against a column of 1e38. The gate rows are small
        /// codes under scale byte 0x01; the up rows are +6 over their first
        /// two blocks and −6 over the last two under scale byte 0x7e (448),
        /// so lanes 0..4 sum to +inf, lanes 4..8 to −inf and the butterfly to
        /// NaN, which the clamp alone would map to a finite value.
        fn overflow(gpu: &Gpu, nv: &Nvfp4SelKernels) -> Result<bool, GateError> {
            const K: usize = 256;
            const ROWS: usize = 8;
            let stream = gpu.stream();
            let block = |scale: u8, codes: u8| {
                let mut b = vec![scale; 4];
                b.extend([codes; 32]);
                b
            };
            let planes = |head: Vec<u8>, tail: Vec<u8>, scale: f32| {
                let row = [head.as_slice(), &head, &tail, &tail].concat();
                let Planes { qs, e } = repack::<Nvfp4>(&row.repeat(ROWS), ROWS, K)?;
                Ok::<_, GateError>((
                    DeviceTensor::upload(stream, &qs, ROWS, K / 8)?,
                    DeviceTensor::upload(stream, &e, ROWS, K / NV_BLOCK_VALUES)?,
                    DeviceTensor::upload(stream, &[scale], 1, 1)?,
                ))
            };
            let (gq, ge, gs) = planes(block(0x01, 0x11), block(0x01, 0x11), 1.0e-37)?;
            let (uq, ue, us) = planes(block(0x7e, 0x77), block(0x7e, 0xff), 1.0)?;
            let x = DeviceBuffer::from_host(stream, &[1.0e38f32; K])?;
            let mut act = Q8Act::with_k(stream, 1, K)?;
            gpu.enqueue_quantize_q8_1_layer(&x, &mut act, 0)?;
            let sel = DeviceBuffer::from_host(stream, &[0u32])?;
            let a = Nvfp4GateUp {
                gate: Nvfp4Stack {
                    qs: &gq,
                    e: &ge,
                    scales: &gs,
                },
                up: Nvfp4Stack {
                    qs: &uq,
                    e: &ue,
                    scales: &us,
                },
                act: &act,
                sel: &sel,
                n_slots: 1,
                rows_per_expert: ROWS,
                slots_per_col: 1,
                rule: Act::SwigluClamp { limit: LIMIT },
            };
            let mut h = DeviceBuffer::from_host(stream, &[SENT; ROWS])?;
            nv.enqueue_gate_up(stream, &a, gpu.unlabelled_sink(), &mut h)?;
            let h = h.to_host_vec(stream)?;
            let named = gpu.take_fault()?.is_some_and(|f| {
                GpuError::fault("gate_nvfp4", f)
                    .to_string()
                    .contains("f32_product")
            });
            let nan = h.iter().all(|v| v.is_nan());
            println!(
                "nvfp4 clause=routed_overflow fault_names_f32_product={named} rows_nan={nan} {}",
                verdict(named && nan)
            );
            Ok(named && nan)
        }

        /// Print one launch's clauses over its outputs `out`, `rows` a slot:
        /// `clean` (no fault, and for the gate·up the card's block scales the
        /// host's), each `HOST` slot's rows at the sentinel, and `bits` and
        /// `band` over every card slot's [`judge`] in `per`.
        fn report(
            name: &str,
            clean: bool,
            per: &[(bool, f64, f64)],
            out: &[f32],
            rows: usize,
        ) -> bool {
            let host = |s: &usize| SEL[*s] == HOST;
            let sentinel = (0..SEL.len())
                .filter(host)
                .all(|s| out[s * rows..][..rows].iter().all(|v| *v == SENT));
            let card: Vec<_> = (0..SEL.len())
                .filter(|s| !host(s))
                .flat_map(|s| per[s * rows..][..rows].iter().copied())
                .collect();
            let (bits, band, worst) = tally(&card);
            println!(
                "nvfp4 clause={name} clean={clean} host_slots_sentinel={sentinel} {}",
                verdict(clean && sentinel)
            );
            println!(
                "nvfp4 clause={name} bits equal={bits}/{} {}",
                card.len(),
                verdict(bits == card.len())
            );
            println!(
                "nvfp4 clause={name} band worst_err_over_band={worst:.3e} {}",
                verdict(band)
            );
            clean && sentinel && bits == card.len() && band
        }
    }

    /// `--bench-kernels` (module doc): one column at each shape, the planes
    /// of [`nvfp4_bytes`] (every scale byte defined, so the repack takes them),
    /// the activation from `fill_pattern_f32`.
    mod bench {
        use super::{ROUTED_FF, ROUTED_N, SHAPES, nvfp4_bytes};
        use crate::bench_arm::{GREPS, N, ROUNDS, burst, fill_pattern_f32, print_arm, replay};
        use bloomery_gpu::kquant::Act;
        use bloomery_gpu::mxfp4::{NV_BLOCK_BYTES, NV_BLOCK_VALUES, Nvfp4, Nvfp4Kernels};
        use bloomery_gpu::mxfp4::{Nvfp4Weight, Planes, repack};
        use bloomery_gpu::mxfp4_sel::{Nvfp4Down, Nvfp4GateUp, Nvfp4SelKernels, Nvfp4Stack};
        use bloomery_gpu::q5::Q8Blocks32;
        use bloomery_gpu::{DeviceTensor, Gpu, Q8Act};
        use bloomery_gpu_gates::GateError;
        use cuda_core::{CudaStream, DeviceBuffer};

        pub(super) const ROUTED_ARMS: [&str; 2] = ["nvfp4_sel_gate_up", "nvfp4_sel_down"];

        /// The experts in the stack, and the slots of a launch's one token.
        const EXPERTS: usize = 128;
        const SLOTS: usize = 8;

        /// The planes and scales of a stack of [`EXPERTS`] of `rows` rows of
        /// `k` values.
        /// An expert stack's code plane, block-scale plane and per-expert scales.
        type Stack = (DeviceTensor<u32>, DeviceTensor<u32>, DeviceTensor<f32>);

        fn stack(
            stream: &CudaStream,
            rows: usize,
            k: usize,
            seed: u64,
        ) -> Result<Stack, GateError> {
            let all = EXPERTS * rows;
            let Planes { qs, e } = repack::<Nvfp4>(&nvfp4_bytes(all, k, seed), all, k)?;
            Ok((
                DeviceTensor::upload(stream, &qs, all, k / 8)?,
                DeviceTensor::upload(stream, &e, all, k / NV_BLOCK_VALUES)?,
                DeviceTensor::upload(stream, &[0.01f32; EXPERTS], EXPERTS, 1)?,
            ))
        }

        /// The routed arms (module doc). The bytes are a
        /// launch's eight experts' planes, the activation's codes and scales,
        /// and the outputs.
        fn routed(gpu: &Gpu, only: Option<&str>) -> Result<(), GateError> {
            let keep = |op: &str| only.is_none_or(|o| op.starts_with(o));
            let [up_op, down_op] = ROUTED_ARMS;
            if !keep(up_op) && !keep(down_op) {
                return Ok(());
            }
            let stream = gpu.stream();
            let nv = Nvfp4SelKernels::load(gpu.context(), gpu.fault_word())?;
            // Launch `i` of a burst reads experts `8(i mod 16)..`.
            let sels = (0..EXPERTS / SLOTS)
                .map(|j| {
                    let ids: Vec<u32> = (0..SLOTS).map(|s| (j * SLOTS + s) as u32).collect();
                    DeviceBuffer::from_host(stream, &ids)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let plane_bytes =
                |rows: usize, k: usize| SLOTS * rows * k / NV_BLOCK_VALUES * NV_BLOCK_BYTES;
            if keep(up_op) {
                let (gq, ge, gs) = stack(stream, ROUTED_FF, ROUTED_N, 0x5eed_0301)?;
                let (uq, ue, us) = stack(stream, ROUTED_FF, ROUTED_N, 0x5eed_0302)?;
                let x = DeviceBuffer::from_host(stream, &fill_pattern_f32(ROUTED_N))?;
                let mut act = Q8Act::with_k(stream, 1, ROUTED_N)?;
                gpu.enqueue_quantize_q8_1_layer(&x, &mut act, 0)?;
                let mut h = DeviceBuffer::<f32>::zeroed(stream, SLOTS * ROUTED_FF)?;
                let sink = gpu.unlabelled_sink();
                let mut i = 0;
                let mut enq = |st: &CudaStream| {
                    let a = Nvfp4GateUp {
                        gate: Nvfp4Stack {
                            qs: &gq,
                            e: &ge,
                            scales: &gs,
                        },
                        up: Nvfp4Stack {
                            qs: &uq,
                            e: &ue,
                            scales: &us,
                        },
                        act: &act,
                        sel: &sels[i % sels.len()],
                        n_slots: SLOTS,
                        rows_per_expert: ROUTED_FF,
                        slots_per_col: SLOTS,
                        rule: Act::SiluMul,
                    };
                    i += 1;
                    nv.enqueue_gate_up(st, &a, sink, &mut h)
                };
                let eager = burst(stream, &mut enq)?;
                let g = gpu.capture(|st| (0..N).try_for_each(|_| enq(st)))?;
                let graph = replay(stream, &g)?;
                let bytes = (2 * plane_bytes(ROUTED_FF, ROUTED_N)
                    + ROUTED_N
                    + 4 * (ROUTED_N / 128)
                    + 4 * SLOTS * ROUTED_FF) as u64;
                print_arm(
                    up_op,
                    SLOTS * ROUTED_FF,
                    bytes,
                    g.node_count(),
                    eager,
                    graph,
                );
            }
            if keep(down_op) {
                let (dq, de, ds) = stack(stream, ROUTED_N, ROUTED_FF, 0x5eed_0303)?;
                let hx = DeviceBuffer::from_host(stream, &fill_pattern_f32(SLOTS * ROUTED_FF))?;
                let mut act = Q8Blocks32::new(stream, ROUTED_FF, SLOTS)?;
                gpu.q5()
                    .enqueue_quantize_q8(stream, &hx, &mut act, gpu.unlabelled_sink())?;
                let mut y = DeviceBuffer::<f32>::zeroed(stream, SLOTS * ROUTED_N)?;
                let sink = gpu.unlabelled_sink();
                let mut i = 0;
                let mut enq = |st: &CudaStream| {
                    let d = Nvfp4Down {
                        w: Nvfp4Stack {
                            qs: &dq,
                            e: &de,
                            scales: &ds,
                        },
                        act: &act,
                        sel: &sels[i % sels.len()],
                        n_slots: SLOTS,
                        rows_per_expert: ROUTED_N,
                        slots_per_col: SLOTS,
                    };
                    i += 1;
                    nv.enqueue_down(st, &d, sink, &mut y)
                };
                let eager = burst(stream, &mut enq)?;
                let g = gpu.capture(|st| (0..N).try_for_each(|_| enq(st)))?;
                let graph = replay(stream, &g)?;
                let bytes = (plane_bytes(ROUTED_N, ROUTED_FF)
                    + SLOTS * (ROUTED_FF + 4 * (ROUTED_FF / 32))
                    + 4 * SLOTS * ROUTED_N) as u64;
                print_arm(
                    down_op,
                    SLOTS * ROUTED_N,
                    bytes,
                    g.node_count(),
                    eager,
                    graph,
                );
            }
            Ok(())
        }

        pub fn run(gpu: &Gpu, only: Option<&str>) -> Result<(), GateError> {
            println!("bench n_per_burst={N} rounds={ROUNDS} graph_launches_per_round={GREPS}");
            let stream = gpu.stream();
            let nv = Nvfp4Kernels::load(gpu.context())?;
            for s in &SHAPES {
                let op = format!("nvfp4_{}", s.rows);
                if only.is_some_and(|o| !op.starts_with(o)) {
                    continue;
                }
                let (rows, k) = (s.rows, s.k);
                let Planes { qs, e } = repack::<Nvfp4>(&nvfp4_bytes(rows, k, s.seed), rows, k)?;
                let qs = DeviceTensor::upload(stream, &qs, rows, k / 8)?;
                let e = DeviceTensor::upload(stream, &e, rows, k / NV_BLOCK_VALUES)?;
                let w = Nvfp4Weight {
                    qs: &qs,
                    e: &e,
                    scale: s.scale,
                };
                let x = DeviceBuffer::from_host(stream, &fill_pattern_f32(k))?;
                let mut act = Q8Act::with_k(stream, 1, k)?;
                gpu.enqueue_quantize_q8_1_layer(&x, &mut act, 0)?;
                let mut y = DeviceBuffer::<f32>::zeroed(stream, rows)?;
                let mut enq = |st: &CudaStream| nv.enqueue_gemv(st, w, &act, &mut y);
                let eager = burst(stream, &mut enq)?;
                let g = gpu.capture(|st| (0..N).try_for_each(|_| enq(st)))?;
                let graph = replay(stream, &g)?;
                // The planes, the column's q4 codes and block scales, the
                // outputs.
                let bytes =
                    (rows * k / NV_BLOCK_VALUES * NV_BLOCK_BYTES + k + 4 * (k / 128) + 4 * rows)
                        as u64;
                print_arm(&op, rows, bytes, g.node_count(), eager, graph);
            }
            routed(gpu, only)
        }
    }
}
