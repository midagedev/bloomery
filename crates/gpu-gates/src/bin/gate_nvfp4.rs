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
//!   (`lane_partial_host::<Nvfp4>` per lane, the warp butterfly, the multiply
//!   by the scale).
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
//! (`bench op=nvfp4_18560`, `bench op=nvfp4_4096`, one column) through
//! `shared/bench.rs`'s burst and replay, timed by `tools/ref/time-gate.sh`
//! under the lease; `--bench-arm <op>` keeps the arms whose name starts with
//! `<op>` (`nvfp4` is both). Nothing there asserts on a result.

#[path = "shared/bench.rs"]
mod bench_arm;

fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_nvfp4", gate::run())
}

mod gate {
    use bloomery_gpu::mxfp4::{NV_BLOCK_BYTES, NV_BLOCK_VALUES, Nvfp4, lane_partial_host};
    use bloomery_gpu::site::{self, KGemvKernels, SiteTy};
    use bloomery_gpu::weights::Weights;
    use bloomery_gpu::{Gpu, Q8Act};
    use bloomery_gpu_gates::act_rule::{Q8_1_BLOCK, q8_1_codes};
    use bloomery_gpu_gates::rounding::{U64, butterfly, gamma};
    use bloomery_gpu_gates::{GateError, GateRng, checks_failed, par_map, verdict};
    use cuda_core::DeviceBuffer;
    use gguf::Split;
    use gguf::quant::{GgmlType, dequant_row};
    use gguf::write::{Layout, TensorDecl, Writer};
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

    /// Write `decls` as a GGUF at `path`.
    fn write_gguf(path: &Path, decls: &[Decl]) -> Result<(), GateError> {
        let tensors = decls
            .iter()
            .map(|(name, dims, ty, bytes)| TensorDecl {
                name: name.clone(),
                dims: dims.clone(),
                type_id: ty.as_u32(),
                nbytes: bytes.len() as u64,
            })
            .collect();
        let layout = Layout::new(&[], tensors).map_err(|e| format!("nvfp4 gguf: {e}"))?;
        let file = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut w = Writer::new(file, layout).map_err(|e| format!("nvfp4 gguf: {e}"))?;
        for (name, _, _, bytes) in decls {
            w.tensor(name, bytes)
                .map_err(|e| format!("nvfp4 gguf: {e}"))?;
        }
        w.finish().map_err(|e| format!("nvfp4 gguf: {e}"))?;
        Ok(())
    }

    /// An NVFP4 weight and its scale as two tensors of a file.
    fn weight_decls(stem: &str, rows: usize, k: usize, bytes: Vec<u8>, scale: f32) -> [Decl; 2] {
        [
            (
                format!("{stem}.weight"),
                vec![k as u64, rows as u64],
                GgmlType::NVFP4,
                bytes,
            ),
            (
                format!("{stem}.scale"),
                vec![1],
                GgmlType::F32,
                scale.to_le_bytes().to_vec(),
            ),
        ]
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        if std::env::args().any(|a| a == "--bench-kernels") {
            return bench::run(&gpu);
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
            decls.extend(weight_decls(s.stem, s.rows, s.k, bytes, s.scale));
        }
        // The refusals' tensors: 8 rows of 256, scale 1 unless named.
        let small = |seed| nvfp4_bytes(8, 256, seed);
        decls.extend(weight_decls("nan", 8, 256, small(7), f32::NAN));
        decls.push(weight_decls("lone", 8, 256, small(8), 1.0)[0].clone());
        let mut undefined = small(9);
        undefined[5 * NV_BLOCK_BYTES + 1] = 0x7F;
        decls.extend(weight_decls("undef", 8, 256, undefined, 1.0));
        write_gguf(&path, &decls)?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let stream = gpu.stream();
        let kk = KGemvKernels::load(gpu)?;
        let mut ok = true;
        for (s, bytes) in SHAPES.iter().zip(&raw) {
            ok &= shape_case(gpu, &kk, &split, s, bytes)?;
        }
        for (stem, want) in [
            ("nan", "not a finite value"),
            ("lone", "lone.scale is not in the file"),
            (
                "undef",
                "row 1 block 1 sub-block 1: scale byte 0x7f is no UE4M3 scale",
            ),
        ] {
            let name = format!("{stem}.weight");
            let got = match Weights::load_where(stream, &split, |n| n == name) {
                Ok(_) => "loaded".to_string(),
                Err(e) => e.to_string(),
            };
            let pass = got.contains(want);
            println!(
                "nvfp4 clause=refusal tensor={name} want={want:?} got={got:?} {}",
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
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
        // Each row dequantized by the host's port: the reference's weights.
        let wf: Vec<Vec<f32>> = par_map(rows, |r| {
            let mut v = vec![0.0f32; k];
            dequant_row(
                GgmlType::NVFP4,
                &bytes[r * row_bytes..][..row_bytes],
                &mut v,
            )
            .unwrap_or_else(|e| panic!("row {r}: {e}"));
            v
        });
        let lane_terms = 2 * (k / 32).div_ceil(32);
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
                let lanes: [f32; 32] =
                    std::array::from_fn(|lane| lane_partial_host::<Nvfp4>(row, q, d, lane));
                let host = butterfly(lanes) * s.scale;
                let (mut sum, mut mag) = (0.0f64, 0.0f64);
                for (sub, wv) in wf[r].as_chunks::<16>().0.iter().enumerate() {
                    let t: f64 = wv
                        .iter()
                        .zip(&q[16 * sub..])
                        .map(|(&wj, &qj)| f64::from(wj) * f64::from(qj))
                        .sum::<f64>()
                        * f64::from(d[16 * sub / Q8_1_BLOCK]);
                    sum += t;
                    mag += t.abs();
                }
                let scale = f64::from(s.scale);
                let reference = scale * sum;
                let band = (gamma(lane_terms + 6) + (k / 16) as f64 * U64) * scale.abs() * mag;
                let got = y[at];
                (
                    got.to_bits() == host.to_bits(),
                    (f64::from(got) - reference).abs(),
                    band,
                )
            });
            let bits = per.iter().all(|p| p.0);
            let bits_n = per.iter().filter(|p| p.0).count();
            let band_ok = per.iter().all(|p| p.1 <= p.2);
            let worst = per
                .iter()
                .map(|p| if p.2 > 0.0 { p.1 / p.2 } else { p.1 })
                .fold(0.0f64, f64::max);
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

    /// `--bench-kernels` (module doc): one column at each shape, the planes
    /// of [`nvfp4_bytes`] (every scale byte defined, so the repack takes them),
    /// the activation from `fill_pattern_f32`.
    mod bench {
        use super::{SHAPES, nvfp4_bytes};
        use crate::bench_arm::{GREPS, N, ROUNDS, burst, fill_pattern_f32, print_arm, replay};
        use bloomery_gpu::mxfp4::{NV_BLOCK_BYTES, NV_BLOCK_VALUES, Nvfp4, Nvfp4Kernels};
        use bloomery_gpu::mxfp4::{Nvfp4Weight, Planes, repack};
        use bloomery_gpu::{DeviceTensor, Gpu, Q8Act};
        use bloomery_gpu_gates::GateError;
        use cuda_core::{CudaStream, DeviceBuffer};

        pub fn run(gpu: &Gpu) -> Result<(), GateError> {
            println!("bench n_per_burst={N} rounds={ROUNDS} graph_launches_per_round={GREPS}");
            let only = {
                let mut args = std::env::args().skip_while(|a| a != "--bench-arm");
                match args.next() {
                    None => None,
                    Some(_) => Some(args.next().ok_or(
                        "gate_nvfp4: --bench-arm wants an arm name (a `bench op=` prefix)",
                    )?),
                }
            };
            let stream = gpu.stream();
            let nv = Nvfp4Kernels::load(gpu.context())?;
            let mut ran = false;
            for s in &SHAPES {
                let op = format!("nvfp4_{}", s.rows);
                if only.as_deref().is_some_and(|o| !op.starts_with(o)) {
                    continue;
                }
                ran = true;
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
            if ran {
                Ok(())
            } else {
                Err(format!("gate_nvfp4: --bench-arm {only:?} names no arm").into())
            }
        }
    }
}
