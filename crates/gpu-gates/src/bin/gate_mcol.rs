//! GPU gate for the m-column kernels: a k-token pass launches the step's
//! gemvs with m = k activation columns, and "k-token step == k one-token
//! steps, bit for bit" holds only if column c of an m-column launch is the
//! m = 1 launch of column c. Pinned here, bit for bit, for every m in 1..=8
//! on real rows of the V4.1 file:
//!
//! - `q8_0_gemv` (the launcher routes m > 1 to `q8_0_gemv_mcol`) at every
//!   Q8_0 site of the chain, plus synthetic rows whose k ends in each tail
//!   of the lane walk; `q8_0_gemv_mcol` in the token-major layout; and
//!   `q8_0_gemv_mcol` at m = 1 against `q8_0_gemv`, the decode kernel;
//! - `q8_0_gemv_heads_mcol` on `attn_output_a` (a head per group) against
//!   `q8_0_gemv_heads`, the decode launch, column by column;
//! - `q6k_gemv` on `output.weight` and `q3k_gemv` on an expert stack's rows,
//!   through the shared q8_1 quantizer (column c's code blocks are its own);
//! - the output head at m rows (`Head::with_m`): each row's normed vector,
//!   logits and argmax token against the m = 1 head on that row alone, and
//!   the captured graph's tokens against eager.
//!
//! The m = 1 launch of a column reads that column uploaded alone, so a
//! column's answer can depend on nothing but its own activations.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_mcol: built without the `gpu` feature; see `just gate-gpu-mcol`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_mcol", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::head::Head;
    use bloomery_gpu::model::{Q8_0GemvHeadsArgs, StepKernels};
    use bloomery_gpu::q8f32::{GemvOut, Q8_0GemvHeadsMcolArgs, Q8_0GemvMcolArgs, Q8F32Kernels};
    use bloomery_gpu::weights::{DevWeight, Weights, q8_0_planes};
    use bloomery_gpu::{DeviceTensor, Gpu, Q8Act};
    use bloomery_gpu_gates::{
        GateError, activations, bits_equal, checks_failed, expect_arch, ref_model_path, verdict,
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

    /// A Q8_0 weight's device planes.
    struct Q8 {
        name: String,
        k: usize,
        rows: usize,
        qs: DeviceTensor<u32>,
        d: DeviceTensor<u16>,
    }

    /// The first `cap` rows of 2-D Q8_0 tensor `name` as device planes;
    /// `None` when the file has no such tensor.
    fn load_q8(
        split: &Split,
        stream: &CudaStream,
        name: &str,
        cap: usize,
    ) -> Result<Option<Q8>, GateError> {
        let Some((s, t)) = split.find(name) else {
            return Ok(None);
        };
        let &[k, rows] = t.dims.as_slice() else {
            return Ok(None);
        };
        if t.ty != GgmlType::Q8_0 {
            return Ok(None);
        }
        let (k, rows) = (usize::try_from(k)?, usize::try_from(rows)?.min(cap));
        let bytes = split.shard(s).ok_or("shard index out of range")?.data(t)?;
        let blocks: Vec<Q8Block> = bytes[..rows * k / 32 * 34]
            .as_chunks::<34>()
            .0
            .iter()
            .map(Q8Block::from_bytes)
            .collect();
        Ok(Some(q8_from_blocks(
            stream,
            name.to_string(),
            k,
            rows,
            &blocks,
        )?))
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

    /// Synthetic Q8_0 rows: any code and any finite scale is a weight, so the
    /// blocks are an LCG's — codes over all of i8, scales f16 in [2^-8, 2).
    fn synthetic_q8(
        stream: &CudaStream,
        k: usize,
        rows: usize,
        seed: u32,
    ) -> Result<Q8, GateError> {
        let mut st = seed;
        let mut next = move || {
            st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            st
        };
        let blocks: Vec<Q8Block> = (0..rows * k / 32)
            .map(|_| {
                let r = next();
                let mut q = [0i8; 32];
                for c in &mut q {
                    *c = (next() >> 24) as u8 as i8;
                }
                Q8Block {
                    d: (((7 + (r >> 16) % 9) << 10) | (r & 0x3ff)) as u16,
                    q,
                }
            })
            .collect();
        q8_from_blocks(stream, "synthetic".to_string(), k, rows, &blocks)
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

    /// One Q8_0 weight: for every m, the routed launch's columns against the
    /// decode kernel on each column, the same in the token-major layout, and
    /// `q8_0_gemv_mcol` at m = 1 against the decode kernel.
    fn check_q8(
        q8: &Q8F32Kernels,
        stream: &CudaStream,
        w: &Q8,
        seed: u32,
    ) -> Result<bool, GateError> {
        let x = activations(w.k, 8, seed);
        let mut y1: Vec<Vec<f32>> = Vec::with_capacity(8);
        for c in 0..8 {
            let xc = DeviceBuffer::from_host(stream, &x[c * w.k..(c + 1) * w.k])?;
            let mut y = DeviceBuffer::from_host(stream, &vec![f32::NAN; w.rows])?;
            q8.enqueue_q8_0_gemv(stream, &w.qs, &w.d, &xc, 1, &mut y)?;
            y1.push(y.to_host_vec(stream)?);
        }
        let x_dev = DeviceBuffer::from_host(stream, &x)?;
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
            let ym = y.to_host_vec(stream)?;
            tm_misses.push(
                (0..m)
                    .map(|c| {
                        (0..w.rows)
                            .filter(|&r| ym[c * w.rows + r].to_bits() != y1[c][r].to_bits())
                            .count()
                    })
                    .sum::<usize>(),
            );
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
            "q8_0 site={} k={} rows={} col_vs_m1_misses[m=1..8]={misses:?} \
             token_major_misses[m=1..8]={tm_misses:?} mcol_m1_eq_decode={mcol1} {}",
            w.name,
            w.k,
            w.rows,
            verdict(pass)
        );
        Ok(pass)
    }

    /// `attn_output_a` through the per-head m-column launch: column c of
    /// every m (token-major y) against the decode launch of column c.
    fn check_heads(
        q8: &Q8F32Kernels,
        step: &StepKernels,
        stream: &CudaStream,
        w: &Q8,
        hp: &Hparams,
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
        let x = activations(width, 8, 41);
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
        let x_dev = DeviceBuffer::from_host(stream, &x)?;
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
            let ym = y.to_host_vec(stream)?;
            misses.push(
                (0..m)
                    .map(|c| {
                        (0..out)
                            .filter(|&r| ym[c * out + r].to_bits() != y1[c][r].to_bits())
                            .count()
                    })
                    .sum::<usize>(),
            );
        }
        let pass = misses.iter().all(|&n| n == 0);
        println!(
            "q8_0_heads site={} groups={groups} rank={rank} group_k={} col_vs_m1_misses[m=1..8]={misses:?} {}",
            w.name,
            w.k,
            verdict(pass)
        );
        Ok(pass)
    }

    /// Which K-quant gemv a word-plane weight goes through.
    #[derive(Clone, Copy)]
    enum Kq {
        Q3k,
        Q6k,
    }

    /// A K-quant word plane through the shared quantizer: column c of every
    /// m against the m = 1 launch of column c.
    fn check_kquant(
        gpu: &Gpu,
        label: &str,
        kind: Kq,
        w: &DeviceTensor<u32>,
        k: usize,
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let rows = w.rows();
        let x = activations(k, 8, 17);
        let run = |xs: &[f32], m: usize| -> Result<Vec<f32>, GateError> {
            let x_dev = DeviceBuffer::from_host(stream, xs)?;
            let mut act = Q8Act::with_k(stream, m, k)?;
            let mut y = DeviceBuffer::from_host(stream, &vec![f32::NAN; rows * m])?;
            gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
            match kind {
                Kq::Q3k => gpu.enqueue_gemv_q3k(w, &act, &mut y)?,
                Kq::Q6k => gpu.enqueue_gemv_q6k(w, &act, &mut y)?,
            }
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
            "kquant site={label} k={k} rows={rows} col_vs_m1_misses[m=1..8]={misses:?} {}",
            verdict(pass)
        );
        Ok(pass)
    }

    /// The first `ROW_CAP` rows of the first expert of the first Q3_K expert
    /// stack whose rows the quantizer can take.
    fn load_q3k(
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
                let words: Vec<u32> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| u32::from_le_bytes(*b))
                    .collect();
                let w = DeviceTensor::upload(stream, &words, rows, row_bytes / 4)?;
                return Ok((name, w, k));
            }
        }
        Err("the file has no Q3_K expert stack with rows of a multiple of 512 values".into())
    }

    /// The head at every m against m one-row heads: normed rows, logits
    /// columns and tokens bit for bit, each token the host argmax of its own
    /// logits column (ties to the lower index), and a captured graph's
    /// tokens equal to eager.
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
            head.capture(gpu, w)?;
            head.set_input(gpu, &x[..m * hidden])?;
            head.launch(gpu)?;
            let graph_ok = head.tokens(gpu)? == tokens;
            let pass = normed_ok && logit_misses == 0 && tokens_ok && host_ok && graph_ok;
            println!(
                "head m={m} normed_rows_eq_m1={normed_ok} logit_misses={logit_misses} tokens={tokens:?} \
                 tokens_eq_m1={tokens_ok} tokens_eq_host_argmax={host_ok} graph_eq_eager={graph_ok} {}",
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

    pub fn run() -> Result<(), GateError> {
        let split = Split::open(ref_model_path()?)?;
        expect_arch(&split, Arch::Deepseek41, "gate-gpu-mcol")?;
        let hp = Hparams::read(&split)?;
        let gpu = Gpu::new()?;
        let stream = gpu.stream();
        let step = StepKernels::load(gpu.context())?;
        let q8 = gpu.q8f32();
        println!(
            "gate_mcol: device {} — n_embd {} layers {} vocab {}",
            gpu.device_name()?,
            hp.n_embd,
            hp.n_layer,
            hp.n_vocab
        );
        let mut ok = true;

        // Every Q8_0 site of the chain, at the first layer that carries it.
        let sites: [fn(usize) -> String; 7] = [
            names::attn_q_a,
            names::attn_q_b,
            names::attn_kv,
            names::attn_output_b,
            names::indexer_attn_q_b,
            names::ffn_down_shexp,
            names::engram_wkv,
        ];
        for (i, site) in sites.iter().enumerate() {
            let found = (0..hp.n_layer)
                .find_map(|l| load_q8(&split, stream, &site(l), ROW_CAP).transpose());
            match found {
                Some(w) => ok &= check_q8(q8, stream, &w?, 100 + i as u32)?,
                None => {
                    eprintln!("FAIL: no layer carries a 2-D Q8_0 {}", site(0));
                    ok = false;
                }
            }
        }
        // Every tail of the lane walk: one word per lane at most (128), whole
        // trips (1024), and trips, a leftover step and a partial word (480,
        // 992); 4099 rows leave the last block short.
        for (k, seed) in [(128usize, 7u32), (480, 9), (992, 11), (1024, 13)] {
            ok &= check_q8(q8, stream, &synthetic_q8(stream, k, ROW_CAP, seed)?, seed)?;
        }

        // Every row: the head geometry is the whole tensor's.
        let whole = |l| load_q8(&split, stream, &names::attn_output_a(l), usize::MAX);
        let wo_a = (0..hp.n_layer)
            .find_map(|l| whole(l).transpose())
            .ok_or("no layer carries a 2-D Q8_0 attn_output_a")??;
        ok &= check_heads(q8, &step, stream, &wo_a, &hp)?;

        let (q3_name, q3, q3_k) = load_q3k(&split, stream, &hp)?;
        ok &= check_kquant(&gpu, &q3_name, Kq::Q3k, &q3, q3_k)?;

        let keep = [names::output(), names::output_norm()];
        let w = Weights::load_where(stream, &split, |n| keep.iter().any(|k| k == n))?;
        let Some(DevWeight::KQuant { w: out_w, k, .. }) = w.get(&names::output()) else {
            return Err(format!(
                "{} is not resident as a K-quant word plane",
                names::output()
            )
            .into());
        };
        ok &= check_kquant(&gpu, &names::output(), Kq::Q6k, out_w, *k)?;
        ok &= check_head(&gpu, &w, hp.rms_eps)?;

        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: column c of every m-column launch (m = 1..8) is the m = 1 launch of column c, \
             bit for bit — q8_0, q8_0 heads, q3_K, q6_K and the head's tokens"
        );
        Ok(())
    }
}
