//! GPU gate for the m-column kernels: a k-token pass launches the step's
//! gemvs with m = k activation columns, and "k-token step == k one-token
//! steps, bit for bit" holds only if column c of an m-column launch is the
//! m = 1 launch of column c. Pinned here, bit for bit, for every m in 1..=8,
//! on whichever V4.1 file the tree runs — every site is picked by the type
//! the file gives it:
//!
//! - the chain's plain projections (`q_a`, `q_b`, `kv`, `wo_b`, the
//!   indexer's and the compressor's, `engram_wkv`, the shared expert's down):
//!   Q8_0 through `q8_0_gemv` (m > 1 routed to `q8_0_gemv_mcol`, also in the
//!   token-major layout and at m = 1 against the decode kernel), Q3_K and
//!   Q4_K through their gemvs behind the shared q8_1 quantizer, Q5_K through
//!   the V4.1 dense f32-activation gemv (`DenseKernels::enqueue_m`);
//! - synthetic Q3_K, Q4_K and Q5_K rows at every K the step has a site at
//!   and each tail of the walks (K in {1280, 2304, 4096, 5120, 8192}: odd
//!   super-block counts, a partial Q4_K iteration, pairs of the Q4_K
//!   prefix), and synthetic Q8_0 rows at each tail of the Q8_0 lane walk;
//! - the q8_1 quantizer: column c of an m-column quantization is that column
//!   quantized alone, every byte of the five planes;
//! - `attn_output_a`'s block diagonal: `q8_0_gemv_heads_mcol` on a Q8_0 file,
//!   `ds41_q3k_gemv_heads_mcol` on a Q3_K one, against the decode launch;
//! - the shared expert's fused Q3_K gate·up·SwiGLU at m tokens;
//! - `q6k_gemv` on `output.weight`, `q3k_gemv` on an expert stack's rows, and
//!   the output head at m rows (`Head::with_m`): each row's normed vector,
//!   logits and argmax token against the m = 1 head on that row alone.
//!
//! Activations: seeded random columns for every site, and for the sites
//! whose input the V4.1 oracle set dumps, the set's own rows (its prefill
//! tokens, cycled to 8 columns). The m = 1 launch of a column reads that
//! column uploaded alone, so a column's answer can depend on nothing but its
//! own activations. Every check prints an `m1=` digest of its m = 1 outputs,
//! so two trees' decode bits compare line by line.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!("gate_mcol: built without the `deepseek41` feature; see `just gate-gpu-mcol`.");
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_mcol", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use bloomery_gpu::fused::{Q8ActHost, readback_q8act};
    use bloomery_gpu::head::Head;
    use bloomery_gpu::model::{Q8_0GemvHeadsArgs, StepKernels};
    use bloomery_gpu::q8f32::{GemvOut, Q8_0GemvHeadsMcolArgs, Q8_0GemvMcolArgs, Q8F32Kernels};
    use bloomery_gpu::weights::{DevWeight, Weights, q8_0_planes};
    use bloomery_gpu::{DeviceTensor, Gpu, Q8Act};
    use bloomery_gpu_deepseek41::dense::{Dense, DenseKernels, Q3kHeadsMcolArgs};
    use bloomery_gpu_gates::{
        Fnv1a64, GateError, RefManifest, V41_SET_FAMILY, activations, bits_equal, checks_failed,
        expect_arch, load_ref_logical_in, ref_dir_named, ref_model_path, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::Split;
    use gguf::quant::{GgmlType, Q8Block};
    use model::arch::Arch;
    use model::arch::deepseek41::hparams::Hparams;
    use model::arch::deepseek41::names;

    /// Column counts a launch takes.
    const MS: std::ops::RangeInclusive<usize> = 1..=8;
    /// Rows kept of a site: enough warps for every block of a launch to be
    /// full and one to be short, without uploading a whole expert stack.
    const ROW_CAP: usize = 4099;
    /// The ik build every V4.1 oracle set is dumped from; a set without it
    /// is stale.
    const SET_BUILD: &str = "db517b69";

    /// The gate's shared handles.
    struct Cx<'a> {
        gpu: &'a Gpu,
        q8: &'a Q8F32Kernels,
        dense: &'a DenseKernels,
    }

    /// A Q8_0 weight's device planes.
    struct Q8 {
        name: String,
        k: usize,
        rows: usize,
        qs: DeviceTensor<u32>,
        d: DeviceTensor<u16>,
    }

    /// A 2-D tensor's first rows: its type, row length, row count and bytes.
    type Rows = (GgmlType, usize, usize, Vec<u8>);

    /// The first `cap` rows of 2-D tensor `name` ([`Rows`]); `None` when the
    /// file has no such 2-D tensor.
    fn load_rows(split: &Split, name: &str, cap: usize) -> Result<Option<Rows>, GateError> {
        let Some((s, t)) = split.find(name) else {
            return Ok(None);
        };
        let &[k, rows] = t.dims.as_slice() else {
            return Ok(None);
        };
        let (k, rows) = (usize::try_from(k)?, usize::try_from(rows)?.min(cap));
        let Some(rb) = row_bytes(t.ty, k) else {
            return Ok(Some((t.ty, k, rows, Vec::new())));
        };
        let bytes = split.shard(s).ok_or("shard index out of range")?.data(t)?;
        Ok(Some((t.ty, k, rows, bytes[..rows * rb].to_vec())))
    }

    /// Bytes of one row of `k` values of the formats this gate runs.
    fn row_bytes(ty: GgmlType, k: usize) -> Option<usize> {
        let (vals, bytes) = match ty {
            GgmlType::Q8_0 => (32, 34),
            GgmlType::Q3_K => (256, 110),
            GgmlType::Q4_K => (256, 144),
            GgmlType::Q5_K => (256, 176),
            _ => return None,
        };
        k.is_multiple_of(vals).then_some(k / vals * bytes)
    }

    fn q8_from_blocks(
        stream: &CudaStream,
        name: String,
        k: usize,
        rows: usize,
        blocks: &[Q8Block],
    ) -> Result<Q8, GateError> {
        let (qs, d) = q8_0_planes(blocks);
        Ok(Q8 {
            qs: DeviceTensor::upload(stream, &qs, rows, k / 4)?,
            d: DeviceTensor::upload(stream, &d, rows, k / 32)?,
            name,
            k,
            rows,
        })
    }

    fn q8_from_bytes(
        stream: &CudaStream,
        name: String,
        k: usize,
        rows: usize,
        bytes: &[u8],
    ) -> Result<Q8, GateError> {
        let blocks: Vec<Q8Block> = bytes
            .as_chunks::<34>()
            .0
            .iter()
            .map(Q8Block::from_bytes)
            .collect();
        q8_from_blocks(stream, name, k, rows, &blocks)
    }

    /// A K-quant weight's rows as the card holds them (`CardFormat::KQuant`):
    /// the rows' byte stream as u32 words, zero-padded at its end to a whole
    /// number of words per row — an odd Q3_K super-block count starts every
    /// other row on a half word.
    fn kquant_words(
        stream: &CudaStream,
        bytes: &[u8],
        rows: usize,
    ) -> Result<DeviceTensor<u32>, GateError> {
        let per_row = bytes.len().div_ceil(4).div_ceil(rows);
        let mut padded = bytes.to_vec();
        padded.resize(4 * rows * per_row, 0);
        let words: Vec<u32> = padded
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        Ok(DeviceTensor::upload(stream, &words, rows, per_row)?)
    }

    /// An LCG's byte stream.
    fn lcg_bytes(n: usize, seed: u32) -> Vec<u8> {
        let mut st = seed;
        (0..n)
            .map(|_| {
                st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (st >> 24) as u8
            })
            .collect()
    }

    /// An f16 in [2^-8, 2) from 16 random bits.
    fn scale_f16(r: u16) -> u16 {
        ((7 + (r >> 10) % 9) << 10) | (r & 0x3ff)
    }

    /// Synthetic Q8_0 rows: any code and any finite scale is a weight, so the
    /// blocks are an LCG's — codes over all of i8, scales f16 in [2^-8, 2).
    fn synthetic_q8(
        stream: &CudaStream,
        k: usize,
        rows: usize,
        seed: u32,
    ) -> Result<Q8, GateError> {
        let mut b = lcg_bytes(rows * k / 32 * 34, seed);
        for blk in b.as_chunks_mut::<34>().0 {
            let d = scale_f16(u16::from_le_bytes([blk[0], blk[1]]));
            blk[..2].copy_from_slice(&d.to_le_bytes());
        }
        q8_from_bytes(stream, "synthetic".to_string(), k, rows, &b)
    }

    /// Synthetic K-quant rows of `ty`: every byte an LCG's, then each
    /// super-block's f16 scale fields set finite, in [2^-8, 2) — any
    /// quant, scale and high-bit byte is a weight.
    fn synthetic_kquant(ty: GgmlType, k: usize, rows: usize, seed: u32) -> Vec<u8> {
        let sb = row_bytes(ty, 256).unwrap_or(1);
        let mut b = lcg_bytes(rows * (k / 256) * sb, seed);
        // The f16 fields: Q3_K's d at +108; Q4_K's and Q5_K's d, dmin at +0.
        let fields: &[usize] = if ty == GgmlType::Q3_K {
            &[108]
        } else {
            &[0, 2]
        };
        for blk in b.chunks_exact_mut(sb) {
            for &at in fields {
                let d = scale_f16(u16::from_le_bytes([blk[at], blk[at + 1]]));
                blk[at..at + 2].copy_from_slice(&d.to_le_bytes());
            }
        }
        b
    }

    /// Column c of `ym` (`y[r*m + c]`) against `y1[c]` (column c's own
    /// `rows` outputs): the count of rows whose bits differ.
    fn column_misses(ym: &[f32], y1: &[Vec<f32>], m: usize) -> usize {
        y1.iter()
            .enumerate()
            .map(|(c, col)| {
                col.iter()
                    .enumerate()
                    .filter(|&(r, v)| v.to_bits() != ym[r * m + c].to_bits())
                    .count()
            })
            .sum()
    }

    /// Token t of `ym` (`y[t*rows + r]`) against `y1[t]`: the count of rows
    /// whose bits differ.
    fn token_misses(ym: &[f32], y1: &[Vec<f32>], rows: usize) -> usize {
        y1.iter()
            .enumerate()
            .map(|(t, col)| {
                (0..rows)
                    .filter(|&r| ym[t * rows + r].to_bits() != col[r].to_bits())
                    .count()
            })
            .sum()
    }

    /// The `m1=` digest of a check's m = 1 outputs, every column in order.
    fn m1_digest(y1: &[Vec<f32>]) -> String {
        let h = y1.iter().fold(Fnv1a64::default(), |h, c| h.f32s(c));
        format!("{:016x}", h.value())
    }

    /// One Q8_0 weight: for every m, the routed launch's columns against the
    /// decode kernel on each column, the same in the token-major layout, and
    /// `q8_0_gemv_mcol` at m = 1 against the decode kernel.
    fn check_q8(
        q8: &Q8F32Kernels,
        stream: &CudaStream,
        w: &Q8,
        x: &[f32],
        act: &str,
    ) -> Result<bool, GateError> {
        let mut y1: Vec<Vec<f32>> = Vec::with_capacity(8);
        for c in 0..8 {
            let xc = DeviceBuffer::from_host(stream, &x[c * w.k..(c + 1) * w.k])?;
            let mut y = DeviceBuffer::from_host(stream, &vec![f32::NAN; w.rows])?;
            q8.enqueue_q8_0_gemv(stream, &w.qs, &w.d, &xc, 1, &mut y)?;
            y1.push(y.to_host_vec(stream)?);
        }
        let x_dev = DeviceBuffer::from_host(stream, x)?;
        let mut misses = Vec::with_capacity(8);
        for m in MS {
            let mut y = DeviceBuffer::from_host(stream, &vec![f32::NAN; w.rows * m])?;
            q8.enqueue_q8_0_gemv(stream, &w.qs, &w.d, &x_dev, m, &mut y)?;
            misses.push(column_misses(&y.to_host_vec(stream)?, &y1[..m], m));
        }
        // The token-major layout: column c's rows at y[c*rows ..].
        let mut tm_misses = Vec::with_capacity(8);
        for m in MS {
            let mut y = DeviceBuffer::from_host(stream, &vec![f32::NAN; w.rows * m])?;
            q8.enqueue_q8_0_gemv_mcol(
                stream,
                Q8_0GemvMcolArgs {
                    qs: &w.qs,
                    d: &w.d,
                    x: &x_dev,
                    m,
                    out: GemvOut::TokenMajor,
                    y: &mut y,
                },
            )?;
            tm_misses.push(token_misses(&y.to_host_vec(stream)?, &y1[..m], w.rows));
        }
        // m = 1 through the m-column entry: the decode kernel's bits.
        let mut y = DeviceBuffer::from_host(stream, &vec![f32::NAN; w.rows])?;
        q8.enqueue_q8_0_gemv_mcol(
            stream,
            Q8_0GemvMcolArgs {
                qs: &w.qs,
                d: &w.d,
                x: &x_dev,
                m: 1,
                out: GemvOut::RowMajor,
                y: &mut y,
            },
        )?;
        let mcol1 = bits_equal(&y.to_host_vec(stream)?, &y1[0]);
        let pass = misses.iter().all(|&n| n == 0) && tm_misses.iter().all(|&n| n == 0) && mcol1;
        println!(
            "q8_0 site={} k={} rows={} act={act} col_vs_m1_misses[m=1..8]={misses:?} \
             token_major_misses[m=1..8]={tm_misses:?} mcol_m1_eq_decode={mcol1} m1={} {}",
            w.name,
            w.k,
            w.rows,
            m1_digest(&y1),
            verdict(pass)
        );
        Ok(pass)
    }

    /// Which K-quant gemv a word-plane weight goes through.
    #[derive(Clone, Copy)]
    enum Kq {
        Q3k,
        Q4k,
        Q6k,
    }

    impl Kq {
        fn name(self) -> &'static str {
            match self {
                Kq::Q3k => "q3_K",
                Kq::Q4k => "q4_K",
                Kq::Q6k => "q6_K",
            }
        }
    }

    /// Whether column c of readback `hm` is column 0 of readback `h1`, every
    /// byte of the five q8_1 planes, in the quantizer's per-column layout of
    /// `n_sb` super-blocks.
    fn q8_col_eq(hm: &Q8ActHost, c: usize, h1: &Q8ActHost, n_sb: usize) -> bool {
        let (q3, q4, q6) = (
            64 * n_sb.div_ceil(2),
            256 * n_sb.div_ceil(4),
            128 * n_sb.div_ceil(2),
        );
        let (s8, d8) = (8 * n_sb, 2 * n_sb);
        hm.q3[c * q3..(c + 1) * q3] == h1.q3[..q3]
            && hm.q4[c * q4..(c + 1) * q4] == h1.q4[..q4]
            && hm.q6[c * q6..(c + 1) * q6] == h1.q6[..q6]
            && hm.s8[c * s8..(c + 1) * s8] == h1.s8[..s8]
            && bits_equal(&hm.d8[c * d8..(c + 1) * d8], &h1.d8[..d8])
    }

    /// A K-quant word plane through the shared quantizer: column c of every
    /// m against the m = 1 launch of column c, and the quantizer's column c
    /// of every m against that column quantized alone.
    fn check_kquant(
        gpu: &Gpu,
        label: &str,
        kind: Kq,
        w: &DeviceTensor<u32>,
        k: usize,
        x: &[f32],
        act: &str,
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let rows = w.rows();
        let run = |xs: &[f32], m: usize| -> Result<(Vec<f32>, Q8ActHost), GateError> {
            let x_dev = DeviceBuffer::from_host(stream, xs)?;
            let mut act = Q8Act::with_k(stream, m, k)?;
            let mut y = DeviceBuffer::from_host(stream, &vec![f32::NAN; rows * m])?;
            gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
            match kind {
                Kq::Q3k => gpu.enqueue_gemv_q3k(w, &act, &mut y)?,
                Kq::Q4k => gpu.enqueue_gemv_q4k(w, &act, &mut y)?,
                Kq::Q6k => gpu.enqueue_gemv_q6k(w, &act, &mut y)?,
            }
            Ok((y.to_host_vec(stream)?, readback_q8act(stream, &act)?))
        };
        let (y1, a1): (Vec<Vec<f32>>, Vec<Q8ActHost>) = (0..8)
            .map(|c| run(&x[c * k..(c + 1) * k], 1))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();
        let n_sb = k / 256;
        let mut misses = Vec::with_capacity(8);
        let mut quant_misses = Vec::with_capacity(8);
        for m in MS {
            let (ym, am) = run(&x[..m * k], m)?;
            misses.push(column_misses(&ym, &y1[..m], m));
            quant_misses.push((0..m).filter(|&c| !q8_col_eq(&am, c, &a1[c], n_sb)).count());
        }
        let pass = misses.iter().all(|&n| n == 0) && quant_misses.iter().all(|&n| n == 0);
        println!(
            "kquant site={label} type={} k={k} rows={rows} act={act} \
             col_vs_m1_misses[m=1..8]={misses:?} q8_1_col_misses[m=1..8]={quant_misses:?} m1={} {}",
            kind.name(),
            m1_digest(&y1),
            verdict(pass)
        );
        Ok(pass)
    }

    /// A Q5_K weight through the V4.1 dense f32-activation gemv: column c of
    /// every m (`y[r*m + c]`) against the one-column launch of column c.
    fn check_q5k(
        cx: &Cx<'_>,
        label: &str,
        w: &DeviceTensor<u32>,
        k: usize,
        x: &[f32],
        act: &str,
    ) -> Result<bool, GateError> {
        let stream = cx.gpu.stream();
        let rows = w.rows();
        let run = |xs: &[f32], m: usize| -> Result<Vec<f32>, GateError> {
            let x_dev = DeviceBuffer::from_host(stream, xs)?;
            let mut y = DeviceBuffer::from_host(stream, &vec![f32::NAN; rows * m])?;
            cx.dense
                .enqueue_m(cx.gpu, Dense::Q5K(w), &x_dev, None, m, &mut y)?;
            Ok(y.to_host_vec(stream)?)
        };
        let y1 = (0..8)
            .map(|c| run(&x[c * k..(c + 1) * k], 1))
            .collect::<Result<Vec<_>, _>>()?;
        let mut misses = Vec::with_capacity(8);
        for m in MS {
            misses.push(column_misses(&run(&x[..m * k], m)?, &y1[..m], m));
        }
        let pass = misses.iter().all(|&n| n == 0);
        println!(
            "kquant site={label} type=q5_K k={k} rows={rows} act={act} \
             col_vs_m1_misses[m=1..8]={misses:?} m1={} {}",
            m1_digest(&y1),
            verdict(pass)
        );
        Ok(pass)
    }

    /// One plain projection of the file at its own type: Q8_0, Q3_K, Q4_K
    /// or Q5_K; `None` for a type no card gemv here runs.
    #[allow(
        clippy::too_many_arguments,
        reason = "a check over one site's weights and its activation (rust-quality R8)"
    )]
    fn check_site(
        cx: &Cx<'_>,
        label: &str,
        ty: GgmlType,
        k: usize,
        rows: usize,
        bytes: &[u8],
        x: &[f32],
        act: &str,
    ) -> Result<Option<bool>, GateError> {
        let stream = cx.gpu.stream();
        Ok(Some(match ty {
            GgmlType::Q8_0 => check_q8(
                cx.q8,
                stream,
                &q8_from_bytes(stream, label.to_string(), k, rows, bytes)?,
                x,
                act,
            )?,
            GgmlType::Q3_K => check_kquant(
                cx.gpu,
                label,
                Kq::Q3k,
                &kquant_words(stream, bytes, rows)?,
                k,
                x,
                act,
            )?,
            GgmlType::Q4_K => check_kquant(
                cx.gpu,
                label,
                Kq::Q4k,
                &kquant_words(stream, bytes, rows)?,
                k,
                x,
                act,
            )?,
            GgmlType::Q5_K => check_q5k(cx, label, &kquant_words(stream, bytes, rows)?, k, x, act)?,
            _ => return Ok(None),
        }))
    }

    /// The oracle set's rows of dump tensor `name` (`k` values a token) as
    /// eight activation columns, its tokens cycled; `None` when the set has
    /// no such tensor.
    fn real_cols(man: &RefManifest, name: &str, k: usize) -> Result<Option<Vec<f32>>, GateError> {
        if man.tensors.iter().all(|r| r.name != name) {
            return Ok(None);
        }
        let (_, v) = load_ref_logical_in(man, name, 0)?;
        if v.is_empty() || !v.len().is_multiple_of(k) {
            return Err(format!("{name}: {} values, not whole tokens of {k}", v.len()).into());
        }
        let tokens = v.len() / k;
        Ok(Some(
            (0..8)
                .flat_map(|c| v[(c % tokens) * k..(c % tokens + 1) * k].iter().copied())
                .collect(),
        ))
    }

    /// `attn_output_a` (Q8_0) through the per-head m-column launch: column c
    /// of every m (token-major y) against the decode launch of column c.
    fn check_heads_q8(
        q8: &Q8F32Kernels,
        step: &StepKernels,
        stream: &CudaStream,
        w: &Q8,
        hp: &Hparams,
        x: &[f32],
        act: &str,
    ) -> Result<bool, GateError> {
        let width = hp.n_head * hp.head_dim;
        let (groups, rank) = (hp.o_groups, hp.o_lora_rank);
        if w.k * groups != width || w.rows != groups * rank {
            return Err(format!(
                "{} is [{}, {}], want [{width}/{groups}, {groups}*{rank}]",
                w.name, w.k, w.rows
            )
            .into());
        }
        let out = groups * rank;
        let mut y1: Vec<Vec<f32>> = Vec::with_capacity(8);
        for c in 0..8 {
            let xc = DeviceBuffer::from_host(stream, &x[c * width..(c + 1) * width])?;
            let mut y = DeviceBuffer::from_host(stream, &vec![f32::NAN; out])?;
            step.enqueue_q8_0_gemv_heads(
                stream,
                Q8_0GemvHeadsArgs {
                    qs: &w.qs,
                    d: &w.d,
                    x: &xc,
                    rows_per_head: rank,
                    x_head_stride: w.k,
                    y_head_stride: rank,
                    y_off: 0,
                    y: &mut y,
                },
            )?;
            y1.push(y.to_host_vec(stream)?);
        }
        let x_dev = DeviceBuffer::from_host(stream, x)?;
        let mut misses = Vec::with_capacity(8);
        for m in MS {
            let mut y = DeviceBuffer::from_host(stream, &vec![f32::NAN; out * m])?;
            q8.enqueue_q8_0_gemv_heads_mcol(
                stream,
                Q8_0GemvHeadsMcolArgs {
                    qs: &w.qs,
                    d: &w.d,
                    x: &x_dev,
                    rows_per_head: rank,
                    x_head_stride: w.k,
                    y_head_stride: rank,
                    y_off: 0,
                    m,
                    x_col_stride: width,
                    y_col_stride: out,
                    y: &mut y,
                },
            )?;
            misses.push(token_misses(&y.to_host_vec(stream)?, &y1[..m], out));
        }
        let pass = misses.iter().all(|&n| n == 0);
        println!(
            "heads site={} type=q8_0 groups={groups} rank={rank} group_k={} act={act} \
             col_vs_m1_misses[m=1..8]={misses:?} m1={} {}",
            w.name,
            w.k,
            m1_digest(&y1),
            verdict(pass)
        );
        Ok(pass)
    }

    /// `attn_output_a` (Q3_K) through `ds41_q3k_gemv_heads_mcol`: token t of
    /// every m (token-major y) against the decode launch on token t's
    /// `groups` q8_1 columns. The m-token activation is the m one-token
    /// activations laid end to end — the quantizer is column-local, which
    /// `check_kquant` pins.
    fn check_heads_q3k(
        cx: &Cx<'_>,
        name: &str,
        w: &DeviceTensor<u32>,
        hp: &Hparams,
        x: &[f32],
        act: &str,
    ) -> Result<bool, GateError> {
        let stream = cx.gpu.stream();
        let width = hp.n_head * hp.head_dim;
        let (groups, rank) = (hp.o_groups, hp.o_lora_rank);
        let (group_k, out) = (width / groups, groups * rank);
        if w.rows() != out {
            return Err(format!("{name} has {} rows, want {groups}*{rank}", w.rows()).into());
        }
        let mut y1: Vec<Vec<f32>> = Vec::with_capacity(8);
        let mut q3s: Vec<Vec<u64>> = Vec::with_capacity(8);
        let mut d8s: Vec<Vec<f32>> = Vec::with_capacity(8);
        for c in 0..8 {
            let xc = DeviceBuffer::from_host(stream, &x[c * width..(c + 1) * width])?;
            let mut a = Q8Act::with_k(stream, groups, group_k)?;
            cx.gpu.enqueue_quantize_q8_1(&xc, &mut a)?;
            let mut y = DeviceBuffer::from_host(stream, &vec![f32::NAN; out])?;
            cx.dense.enqueue_q3k_heads(stream, w, &a, rank, &mut y)?;
            y1.push(y.to_host_vec(stream)?);
            q3s.push(a.q3().to_host_vec(stream)?);
            d8s.push(a.d8().to_host_vec(stream)?);
        }
        let mut misses = Vec::with_capacity(8);
        for m in MS {
            let q3 = DeviceBuffer::from_host(stream, &q3s[..m].concat())?;
            let d8 = DeviceBuffer::from_host(stream, &d8s[..m].concat())?;
            let mut y = DeviceBuffer::from_host(stream, &vec![f32::NAN; out * m])?;
            cx.dense.enqueue_q3k_heads_mcol(
                stream,
                Q3kHeadsMcolArgs {
                    w,
                    q3: &q3,
                    d8: &d8,
                    n_sb: group_k / 256,
                    groups,
                    rows_per_head: rank,
                    m,
                    y: &mut y,
                },
            )?;
            misses.push(token_misses(&y.to_host_vec(stream)?, &y1[..m], out));
        }
        let pass = misses.iter().all(|&n| n == 0);
        println!(
            "heads site={name} type=q3_K groups={groups} rank={rank} group_k={group_k} act={act} \
             col_vs_m1_misses[m=1..8]={misses:?} m1={} {}",
            m1_digest(&y1),
            verdict(pass)
        );
        Ok(pass)
    }

    /// The shared expert's fused Q3_K gate·up·SwiGLU at m tokens (token-major
    /// h) against the one-token launch on each token.
    #[allow(
        clippy::too_many_arguments,
        reason = "a check over one site's weights and its activation (rust-quality R8)"
    )]
    fn check_shexp_q3k(
        cx: &Cx<'_>,
        label: &str,
        gate: &DeviceTensor<u32>,
        up: &DeviceTensor<u32>,
        k: usize,
        limit: f32,
        x: &[f32],
        act: &str,
    ) -> Result<bool, GateError> {
        let stream = cx.gpu.stream();
        let rows = gate.rows();
        let run = |xs: &[f32], m: usize| -> Result<Vec<f32>, GateError> {
            let x_dev = DeviceBuffer::from_host(stream, xs)?;
            let mut a = Q8Act::with_k(stream, m, k)?;
            cx.gpu.enqueue_quantize_q8_1(&x_dev, &mut a)?;
            let mut h = DeviceBuffer::from_host(stream, &vec![f32::NAN; rows * m])?;
            cx.dense
                .enqueue_shexp_gate_up_q3k(stream, gate, up, &a, limit, &mut h)?;
            Ok(h.to_host_vec(stream)?)
        };
        let y1 = (0..8)
            .map(|c| run(&x[c * k..(c + 1) * k], 1))
            .collect::<Result<Vec<_>, _>>()?;
        let mut misses = Vec::with_capacity(8);
        for m in MS {
            misses.push(token_misses(&run(&x[..m * k], m)?, &y1[..m], rows));
        }
        let pass = misses.iter().all(|&n| n == 0);
        println!(
            "shexp site={label} type=q3_K k={k} rows={rows} limit={limit} act={act} \
             col_vs_m1_misses[m=1..8]={misses:?} m1={} {}",
            m1_digest(&y1),
            verdict(pass)
        );
        Ok(pass)
    }

    /// The first `ROW_CAP` rows of the first expert of the first Q3_K expert
    /// stack whose rows the quantizer can take.
    fn load_q3k_experts(
        split: &Split,
        stream: &CudaStream,
        hp: &Hparams,
    ) -> Result<(String, DeviceTensor<u32>, usize), GateError> {
        for l in 0..hp.n_layer {
            for name in [
                names::ffn_gate_exps(l),
                names::ffn_up_exps(l),
                names::ffn_down_exps(l),
            ] {
                let Some((s, t)) = split.find(&name) else {
                    continue;
                };
                let k = usize::try_from(t.dims[0])?;
                if t.ty != GgmlType::Q3_K || !k.is_multiple_of(512) {
                    continue;
                }
                let row_bytes = 110 * k / 256;
                let rows = usize::try_from(t.dims[1])?.min(ROW_CAP);
                let bytes =
                    &split.shard(s).ok_or("shard index out of range")?.data(t)?[..rows * row_bytes];
                return Ok((name, kquant_words(stream, bytes, rows)?, k));
            }
        }
        Err("the file has no Q3_K expert stack with rows of a multiple of 512 values".into())
    }

    /// The head at every m against m one-row heads: normed rows, logits
    /// columns and tokens bit for bit, each token the host argmax of its own
    /// logits column (ties to the lower index).
    fn check_head(gpu: &Gpu, w: &Weights, eps: f32) -> Result<bool, GateError> {
        let mut one = Head::with_m(gpu, w, eps, 1)?;
        let (hidden, n_vocab) = (one.hidden(), one.n_vocab());
        let x = activations(hidden, 8, 23);
        let mut normed1 = Vec::with_capacity(8);
        let mut logits1 = Vec::with_capacity(8);
        let mut tokens1 = Vec::with_capacity(8);
        for c in 0..8 {
            one.set_input(gpu, &x[c * hidden..(c + 1) * hidden])?;
            one.enqueue(gpu, w)?;
            normed1.push(one.normed_to_host(gpu)?);
            logits1.push(one.logits_to_host(gpu)?);
            tokens1.push(one.token(gpu)?);
        }
        let mut ok = true;
        for m in MS {
            let mut head = Head::with_m(gpu, w, eps, m)?;
            head.set_input(gpu, &x[..m * hidden])?;
            head.enqueue(gpu, w)?;
            let normed = head.normed_to_host(gpu)?;
            let logits = head.logits_to_host(gpu)?;
            let tokens = head.tokens(gpu)?;
            let normed_ok =
                (0..m).all(|c| bits_equal(&normed[c * hidden..(c + 1) * hidden], &normed1[c]));
            let logit_misses = column_misses(&logits, &logits1[..m], m);
            let tokens_ok = tokens[..] == tokens1[..m];
            let host_ok = (0..m).all(|c| {
                let col: Vec<f32> = (0..n_vocab).map(|v| logits[v * m + c]).collect();
                argmax_low(&col) == tokens[c] as usize
            });
            let pass = normed_ok && logit_misses == 0 && tokens_ok && host_ok;
            println!(
                "head m={m} normed_rows_eq_m1={normed_ok} logit_misses={logit_misses} tokens={tokens:?} \
                 tokens_eq_m1={tokens_ok} tokens_eq_host_argmax={host_ok} {}",
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    /// Host argmax, ties to the lower index — the kernel's rule.
    fn argmax_low(v: &[f32]) -> usize {
        let mut best = 0usize;
        for (i, &x) in v.iter().enumerate() {
            if x > v[best] {
                best = i;
            }
        }
        best
    }

    /// A plain projection of the chain: its tensor name at layer `l`, and the
    /// dump tensor at layer `l` that is its input, when the oracle set names
    /// one.
    struct Site {
        name: fn(usize) -> String,
        input: Option<fn(usize) -> String>,
    }

    fn site(name: fn(usize) -> String, input: Option<fn(usize) -> String>) -> Site {
        Site { name, input }
    }

    /// The chain's plain projections.
    fn sites() -> [Site; 12] {
        [
            site(names::attn_q_a, Some(|l| format!("attn_norm-{l}"))),
            site(names::attn_kv, Some(|l| format!("attn_norm-{l}"))),
            site(names::attn_q_b, Some(|l| format!("qr_norm-{l}"))),
            site(
                names::attn_output_b,
                Some(|l| format!("attn_wo_a-{l} (permuted) (cont)")),
            ),
            site(names::ffn_down_shexp, Some(|l| format!("ffn_up_gate-{l}"))),
            site(names::indexer_attn_q_b, None),
            site(names::indexer_proj, None),
            site(names::indexer_attn_k, None),
            site(names::attn_compressor_kv, None),
            site(names::attn_compressor_gate, None),
            site(names::engram_wkv, None),
            site(names::ffn_gate_shexp, None),
        ]
    }

    /// The first layer carrying `name` at each type it has in the file, in
    /// type order of first appearance.
    fn layers_by_type(split: &Split, hp: &Hparams, name: fn(usize) -> String) -> Vec<usize> {
        let mut seen: Vec<GgmlType> = Vec::new();
        let mut out = Vec::new();
        for l in 0..hp.n_layer {
            if let Some((_, t)) = split.find(&name(l))
                && t.dims.len() == 2
                && !seen.contains(&t.ty)
            {
                seen.push(t.ty);
                out.push(l);
            }
        }
        out
    }

    pub fn run() -> Result<(), GateError> {
        let path = ref_model_path()?;
        let split = Split::open(&path)?;
        expect_arch(&split, Arch::Deepseek41, "gate-gpu-mcol")?;
        let hp = Hparams::read(&split)?;
        let gpu = Gpu::new()?;
        let stream = gpu.stream();
        let step = StepKernels::load(gpu.context())?;
        let dense = DenseKernels::load(gpu.context())?;
        let cx = Cx {
            gpu: &gpu,
            q8: gpu.q8f32(),
            dense: &dense,
        };
        let man = RefManifest::read(&ref_dir_named(V41_SET_FAMILY))?;
        let set_ok = man.build.as_deref() == Some(SET_BUILD) && man.header.model() == path.to_str();
        println!(
            "gate_mcol: device {} — {} — n_embd {} layers {} vocab {}; oracle set {} build {:?} \
             model {:?} {}",
            gpu.device_name()?,
            path.display(),
            hp.n_embd,
            hp.n_layer,
            hp.n_vocab,
            man.dir.display(),
            man.build,
            man.header.model(),
            verdict(set_ok)
        );
        let mut ok = set_ok;

        // Every plain projection of the chain, at the first layer of each type
        // the file gives it: random columns, then the set's rows of its input.
        for (i, site) in sites().iter().enumerate() {
            let layers = layers_by_type(&split, &hp, site.name);
            if layers.is_empty() {
                println!("site {}: not in this file", (site.name)(0));
            }
            for l in layers {
                let name = (site.name)(l);
                let (ty, k, rows, bytes) =
                    load_rows(&split, &name, ROW_CAP)?.ok_or("a 2-D site vanished")?;
                let x = activations(k, 8, 100 + i as u32);
                let Some(pass) = check_site(&cx, &name, ty, k, rows, &bytes, &x, "random")? else {
                    println!("site {name}: {ty:?} runs no card gemv here, skipped");
                    continue;
                };
                ok &= pass;
                if let Some(input) = site.input {
                    let dump = input(l);
                    match real_cols(&man, &dump, k)? {
                        Some(x) => {
                            let act = format!("dump:{dump}");
                            ok &= check_site(&cx, &name, ty, k, rows, &bytes, &x, &act)?
                                .unwrap_or(false);
                        }
                        None => {
                            eprintln!("FAIL: the oracle set has no {dump}, {name}'s input");
                            ok = false;
                        }
                    }
                }
            }
        }

        // Synthetic K-quant rows at every K the step has a site at and each
        // tail of the walks: Q3_K's odd super-block counts, Q4_K's partial
        // final iteration (2304, 1280) and pairs of its prefix (4096, 5120,
        // 8192); 4099 rows leave the last block short.
        for (j, ty) in [GgmlType::Q3_K, GgmlType::Q4_K, GgmlType::Q5_K]
            .into_iter()
            .enumerate()
        {
            for (i, k) in [1280usize, 2304, 4096, 5120, 8192].into_iter().enumerate() {
                let seed = 1000 + 10 * j as u32 + i as u32;
                let bytes = synthetic_kquant(ty, k, ROW_CAP, seed);
                let x = activations(k, 8, seed);
                ok &= check_site(&cx, "synthetic", ty, k, ROW_CAP, &bytes, &x, "random")?
                    .unwrap_or(false);
            }
        }
        // Every tail of the Q8_0 lane walk: one word per lane at most (128),
        // whole trips (1024), and trips, a leftover step and a partial word
        // (480, 992).
        for (k, seed) in [(128usize, 7u32), (480, 9), (992, 11), (1024, 13)] {
            let w = synthetic_q8(stream, k, ROW_CAP, seed)?;
            ok &= check_q8(cx.q8, stream, &w, &activations(k, 8, seed), "random")?;
        }

        // The block diagonal: every row, the head geometry is the whole
        // tensor's.
        let wo_a_layer = (0..hp.n_layer)
            .find(|&l| split.find(&names::attn_output_a(l)).is_some())
            .ok_or("no layer carries attn_output_a")?;
        let wo_a = names::attn_output_a(wo_a_layer);
        let (ty, k, rows, bytes) =
            load_rows(&split, &wo_a, usize::MAX)?.ok_or("attn_output_a is not 2-D")?;
        let width = hp.n_head * hp.head_dim;
        let mut heads_x = vec![("random".to_string(), activations(width, 8, 41))];
        let dump = format!("attn-{wo_a_layer}");
        match real_cols(&man, &dump, width)? {
            Some(x) => heads_x.push((format!("dump:{dump}"), x)),
            None => {
                eprintln!("FAIL: the oracle set has no {dump}, {wo_a}'s input");
                ok = false;
            }
        }
        for (act, x) in &heads_x {
            ok &= match ty {
                GgmlType::Q8_0 => {
                    let w = q8_from_bytes(stream, wo_a.clone(), k, rows, &bytes)?;
                    check_heads_q8(cx.q8, &step, stream, &w, &hp, x, act)?
                }
                GgmlType::Q3_K => {
                    let w = kquant_words(stream, &bytes, rows)?;
                    check_heads_q3k(&cx, &wo_a, &w, &hp, x, act)?
                }
                _ => {
                    eprintln!("FAIL: {wo_a} is {ty:?}, want Q8_0 or Q3_K");
                    false
                }
            };
        }

        // The shared expert's fused gate·up, where the file's gate and up are
        // Q3_K (every row).
        let shexp = (0..hp.n_layer).find(|&l| {
            [names::ffn_gate_shexp(l), names::ffn_up_shexp(l)]
                .iter()
                .all(|n| split.find(n).is_some_and(|(_, t)| t.ty == GgmlType::Q3_K))
        });
        match shexp {
            Some(l) => {
                let (g, u) = (names::ffn_gate_shexp(l), names::ffn_up_shexp(l));
                let (_, k, rows, gb) = load_rows(&split, &g, usize::MAX)?.ok_or("gate")?;
                let (_, _, _, ub) = load_rows(&split, &u, usize::MAX)?.ok_or("up")?;
                let (gw, uw) = (
                    kquant_words(stream, &gb, rows)?,
                    kquant_words(stream, &ub, rows)?,
                );
                let limit = hp.layers[l].swiglu_limit_shared;
                let mut xs = vec![("random".to_string(), activations(k, 8, 57))];
                let dump = format!("ffn_norm-{l}");
                match real_cols(&man, &dump, k)? {
                    Some(x) => xs.push((format!("dump:{dump}"), x)),
                    None => {
                        eprintln!("FAIL: the oracle set has no {dump}, {g}'s input");
                        ok = false;
                    }
                }
                for (act, x) in &xs {
                    ok &= check_shexp_q3k(&cx, &g, &gw, &uw, k, limit, x, act)?;
                }
            }
            None => println!("shexp: no layer with Q3_K gate and up in this file"),
        }

        let (q3_name, q3, q3_k) = load_q3k_experts(&split, stream, &hp)?;
        ok &= check_kquant(
            &gpu,
            &q3_name,
            Kq::Q3k,
            &q3,
            q3_k,
            &activations(q3_k, 8, 17),
            "random",
        )?;

        let keep = [names::output(), names::output_norm()];
        let w = Weights::load_where(stream, &split, |n| keep.iter().any(|k| k == n))?;
        let Some(DevWeight::KQuant { w: out_w, k, .. }) = w.get(&names::output()) else {
            return Err(format!(
                "{} is not resident as a K-quant word plane",
                names::output()
            )
            .into());
        };
        ok &= check_kquant(
            &gpu,
            &names::output(),
            Kq::Q6k,
            out_w,
            *k,
            &activations(*k, 8, 17),
            "random",
        )?;
        ok &= check_head(&gpu, &w, hp.rms_eps)?;

        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: column c of every m-column launch (m = 1..8) is the m = 1 launch of column c, \
             bit for bit — q8_0, q3_K, q4_K, q5_K, q6_K, the q8_1 quantizer, the block diagonal, \
             the shared expert's gate·up and the head's tokens"
        );
        Ok(())
    }
}
