//! GPU kernel gate for package P4 (docs/gpu-design.md 작업 꾸러미): the
//! element-wise and reduction kernels of `bloomery_gpu::elem` — embedding row
//! dequant, rms_norm, rope, swiglu, residual add, routed-expert weighted sum,
//! argmax. Two layers per op (the package's gate rule):
//! - **Asserted**: device output vs a host scalar reference computed in this
//!   binary from the SAME input, f64 where a sum is involved, mirroring the
//!   CPU engine's op order per op (cited at each reference fn). Band: exact
//!   bits where the op is exact by construction (embed, add, argmax index);
//!   1e-5 for rms_norm and weighted_sum (reduction tree vs f64 serial);
//!   1e-6 for rope and swiglu (plain f32 ops; swiglu's device `expf` vs the
//!   host's is the distance, printed). Plus a bit-identical rerun per shape.
//! - **Printed, never asserted**: `ik_rel` — the distance to ik CUDA's own
//!   dumped output for the same real input (the block-layer bands are pinned
//!   from these numbers by the lead, not chosen here). `swiglu` has no
//!   isolated oracle row (the reference fuses it with the matmul) and
//!   `argmax` compares indices, so those print `ik_rel=n/a`.
//! Real inputs come from the CUDA oracle dump wherever it holds the op's
//! input; synthetic `activations()` only for shapes the dump cannot supply
//! (swiglu). Every chain is proven from MANIFEST.tsv (dims + op) before use;
//! the residual-add operand pairs are proven by element sums. One captured
//! graph (embed -> rms_norm -> argmax) must replay byte-identically to the
//! eager sequence.
//!
//! Dump convention this gate relies on, probe-verified on the box: VIEW
//! tensors are dumped as flat memory from the view's base pointer, not
//! materialized — `kv_compressed-L.0` is `kv_rope_compressed-L[0..]`,
//! `k_rope-L.0` is `[latent..]`, `q_rope-L.0` is `q-L[nope..]`. The flat
//! property is asserted before each use (`view_flat`); the kernel INPUTS
//! are the logical tensors, read through `ref_tensor_logical` (the v2
//! `.logical.f32` twins).

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_p4: built without the `gpu` feature; see `just gate-gpu-p4`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
use bloomery_gpu_gates::RefRow;
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::{
    activations, bytes_to_words, f32_tensor, find_ref_row, max_rel_err, open_model, ref_manifest,
    ref_tensor_logical, ref_tensor_of, row_bytes, tensor_bytes, view_flat,
};
#[cfg(feature = "gpu")]
use cuda_core::DeviceBuffer;
#[cfg(feature = "gpu")]
use gguf::quant::{GgmlType, dequant_row};

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use bloomery_gpu::elem::ElemKernels;
    use bloomery_gpu::{DeviceTensor, Gpu, GpuModel};

    // Reduction ops' band vs the f64 host reference (the package's gate rule:
    // 1e-5 for a tree-vs-serial sum); 1e-6 for the plain-op band of rope and
    // swiglu where only last ulps can move.
    const REDUCE_BAND: f32 = 1e-5;
    const PLAIN_BAND: f32 = 1e-6;
    // The six-token prompt the CUDA dump was made for (positions 0..5).
    const PROMPT: [u32; 6] = [100000, 549, 6077, 280, 7239, 317];
    // The oracle's own sampled token for `result_output` (the dump's argmax).
    const IK_ARGMAX: u32 = 8913;

    let mut ok = true;
    let gguf = open_model()?;
    let gpu = Gpu::new()?;
    let elem = ElemKernels::load(gpu.context())?;
    let stream = gpu.stream();
    let man = ref_manifest()?;

    // The one architecture-wide rms epsilon the CPU engine reads for every
    // norm (attn_norm, ffn_norm, kv_a norm, output_norm alike) — the key
    // `forward::rms_eps` reads, never a literal.
    let eps: f32 = gguf
        .architecture()
        .and_then(|a| gguf.value(&format!("{a}.attention.layer_norm_rms_epsilon")))
        .and_then(gguf::Value::as_f32)
        .ok_or("gate_p4: metadata <arch>.attention.layer_norm_rms_epsilon missing")?;

    // The rope parameters, built by the CPU engine's own reader — the only
    // public constructor of `RopeParams` reachable from this package (via
    // `GpuModel`); the YaRN cache must be the engine's exact math.
    let mla = GpuModel::load(&gguf, 1)?.mla().clone();

    // The embedding table, uploaded once in the load-time format.
    let (emb_info, emb_bytes) = tensor_bytes(&gguf, "token_embd.weight")?;
    assert_eq!(emb_info.ty, GgmlType::Q3_K, "token_embd.weight type");
    assert_eq!(emb_info.dims[0], 2048, "token_embd.weight K");
    let vocab = emb_info.dims[1] as usize;
    let emb_words = bytes_to_words(emb_bytes);
    assert!(
        vocab > 0 && emb_words.len() % vocab == 0 && emb_words.len() / vocab == 220,
        "token_embd.weight: {} words over {vocab} rows",
        emb_words.len()
    );
    let emb_dev = DeviceTensor::upload(stream, &emb_words, vocab, 220)?;
    let m = PROMPT.len();
    let ids_dev = DeviceBuffer::from_host(stream, &PROMPT)?;

    // ============================================================ embed
    {
        let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, m * 2048)?;
        elem.enqueue_embed_rows(stream, &emb_dev, &ids_dev, &mut y_dev)?;
        stream.synchronize()?;
        let y = y_dev.to_host_vec(stream)?;
        elem.enqueue_embed_rows(stream, &emb_dev, &ids_dev, &mut y_dev)?;
        stream.synchronize()?;
        let rerun = bits_equal(&y, &y_dev.to_host_vec(stream)?);

        // Host reference: the scalar dequantizer on the same row bytes
        // (`forward::embed_with` runs dequant_row per id).
        let rb = row_bytes(GgmlType::Q3_K, 2048)?;
        let mut y_ref = vec![0.0f32; m * 2048];
        for (t, &id) in PROMPT.iter().enumerate() {
            dequant_row(
                GgmlType::Q3_K,
                &emb_bytes[id as usize * rb..][..rb],
                &mut y_ref[t * 2048..(t + 1) * 2048],
            )?;
        }
        let rel = max_rel_err(&y, &y_ref)?;
        let exact = bits_equal(&y, &y_ref);

        // Oracle: inp_embd (GET_ROWS of the same prompt).
        let row = find_ref_row(&man, "inp_embd", 0)?;
        expect(row, "inp_embd", "f32", [2048, 6, 1, 1], "GET_ROWS")?;
        let ik = ref_tensor_of(row)?;
        let ik_rel = max_rel_err(&y, &ik)?;

        let pass = exact && rel == 0.0 && rerun;
        println!(
            "shape op=embed ty=q3_K K=2048 vocab={vocab} m={m} max_rel_err={rel:.3e} bit_exact_host={exact} bit_identical_rerun={rerun} ik_rel={ik_rel:.3e} {}",
            verdict(pass)
        );
        if !pass {
            ok = false;
        }
    }

    // ========================================================== rms_norm
    // (label, input, output, out_occ, gain tensor). Each output row's op is
    // FUSED_RMS_NORM and its dims [K, m] match the input's, which proves the
    // chain. result_norm's input is dumped for the sampled position only
    // (m=1) — checked, not assumed.
    let rms_chains: [(&str, &str, &str, u32, &str); 4] = [
        (
            "attn_norm-0",
            "inp_embd",
            "attn_norm-0",
            0,
            "blk.0.attn_norm.weight",
        ),
        (
            "attn_norm-1",
            "l_out-0",
            "attn_norm-1",
            0,
            "blk.1.attn_norm.weight",
        ),
        (
            "ffn_norm-1",
            "ffn_inp-1",
            "ffn_norm-1",
            0,
            "blk.1.ffn_norm.weight",
        ),
        (
            "result_norm",
            "l_out-26",
            "result_norm",
            0,
            "output_norm.weight",
        ),
    ];
    for (label, in_name, out_name, out_occ, gain_name) in rms_chains {
        let (in_row, x) = load_ref(&man, in_name, 0)?;
        let out_row = find_ref_row(&man, out_name, out_occ)?;
        let k = in_row.ne[0] as usize;
        let mi = in_row.ne[1] as usize;
        expect(&in_row, in_name, "f32", [k as u64, mi as u64, 1, 1], "in")?;
        expect(
            &out_row,
            out_name,
            "f32",
            [k as u64, mi as u64, 1, 1],
            "FUSED_RMS_NORM",
        )?;
        let pass = rms_case(&elem, stream, &x, k, mi, &gguf, gain_name, out_row, eps)?;
        println!(
            "shape op=rms_norm site={label} K={k} m={mi} gain={gain_name} {}",
            pass.1
        );
        if !pass.0 {
            ok = false;
        }
    }
    // The kv_a norm: the input is the latent slice of kv_rope_compressed
    // (the dump's kv_compressed.0 VIEW is the flat base memory — proved
    // below — so the input is the row's logical twin).
    {
        let label = "kv_compressed-1";
        let (base_row, krc) = load_ref(&man, "kv_rope_compressed-1", 0)?;
        let kv_w = mla.latent + mla.rope_dims;
        expect(
            &base_row,
            "kv_rope_compressed-1",
            "f32",
            [kv_w as u64, m as u64, 1, 1],
            "MUL_MAT",
        )?;
        let (view_row, view) = load_ref(&man, label, 0)?;
        expect(
            &view_row,
            label,
            "f32",
            [mla.latent as u64, m as u64, 1, 1],
            "VIEW",
        )?;
        view_flat(&view, &krc, 0, label)?;
        let x = ref_tensor_logical(label, 0)?.1;
        let out_row = find_ref_row(&man, label, 1)?;
        expect(
            &out_row,
            label,
            "f32",
            [mla.latent as u64, m as u64, 1, 1],
            "FUSED_RMS_NORM",
        )?;
        let pass = rms_case(
            &elem,
            stream,
            &x,
            mla.latent,
            m,
            &gguf,
            "blk.1.attn_kv_a_norm.weight",
            out_row,
            eps,
        )?;
        println!(
            "shape op=rms_norm site={label} K={} m={m} gain=blk.1.attn_kv_a_norm.weight {}",
            mla.latent, pass.1
        );
        if !pass.0 {
            ok = false;
        }
    }

    // ============================================================== rope
    // q_rope-L occurrence 0 (VIEW) -> occurrence 1 (ROPE), same for k_rope-L;
    // positions 0..5, one host YaRN cache per position. The VIEW inputs are
    // flat base memory in the dump (proved per layer); the kernel inputs are
    // the rows' logical twins — the rope slice of each q-L head and the rope
    // tail of each token's kv_rope_compressed-L.
    {
        let nd = mla.rope_dims;
        assert_eq!(nd % 2, 0, "rope dims must pair");
        let mut cs = vec![0.0f32; m * nd];
        let mut buf: Vec<f32> = Vec::new();
        for t in 0..m {
            mla.rope.cache_into(t as u32, &mut buf);
            cs[t * nd..(t + 1) * nd].copy_from_slice(&buf);
        }
        let cs_dev = DeviceBuffer::from_host(stream, &cs)?;
        for l in [0usize, 1, 26] {
            // The q side's base tensor, loaded for the flat-view proof.
            let q_name = format!("q-{l}");
            let (q_row, qv) = load_ref(&man, &q_name, 0)?;
            expect(
                &q_row,
                &q_name,
                "f32",
                [(mla.n_head * mla.kq_head) as u64, m as u64, 1, 1],
                "MUL_MAT",
            )?;
            // The k side's base tensor, same proof.
            let krc_name = format!("kv_rope_compressed-{l}");
            let (krc_row, krc) = load_ref(&man, &krc_name, 0)?;
            let kv_w = mla.latent + nd;
            expect(
                &krc_row,
                &krc_name,
                "f32",
                [kv_w as u64, m as u64, 1, 1],
                "MUL_MAT",
            )?;
            for (what, n_vec) in [("q_rope", mla.n_head as u32), ("k_rope", 1u32)] {
                let name = format!("{what}-{l}");
                let (in_row, view) = load_ref(&man, &name, 0)?;
                let out_row = find_ref_row(&man, &name, 1)?;
                expect(
                    &in_row,
                    &name,
                    "f32",
                    [nd as u64, n_vec as u64, m as u64, 1],
                    "VIEW",
                )?;
                expect(
                    &out_row,
                    &name,
                    "f32",
                    [nd as u64, n_vec as u64, m as u64, 1],
                    "ROPE",
                )?;
                // Prove the flat-view convention: the dump equals the base
                // memory from the slice's start, which is NOT the logical
                // tensor fed to the kernel below.
                let base_off = if n_vec == 1 { mla.latent } else { mla.nope };
                view_flat(&view, if n_vec == 1 { &krc } else { &qv }, base_off, &name)?;
                let src = ref_tensor_logical(&name, 0)?.1;

                let src_dev = DeviceBuffer::from_host(stream, &src)?;
                let mut dst_dev = DeviceBuffer::<f32>::zeroed(stream, src.len())?;
                elem.enqueue_rope(stream, &src_dev, &cs_dev, nd, n_vec, m, &mut dst_dev)?;
                stream.synchronize()?;
                let y = dst_dev.to_host_vec(stream)?;
                elem.enqueue_rope(stream, &src_dev, &cs_dev, nd, n_vec, m, &mut dst_dev)?;
                stream.synchronize()?;
                let rerun = bits_equal(&y, &dst_dev.to_host_vec(stream)?);

                let y_ref = rope_ref(&src, &cs, nd, n_vec as usize, m);
                let rel = max_rel_err(&y, &y_ref)?;
                let exact = bits_equal(&y, &y_ref);
                let ik = ref_tensor_of(out_row)?;
                let ik_rel = max_rel_err(&y, &ik)?;
                let pass = rel <= PLAIN_BAND && rerun;
                println!(
                    "shape op=rope site={name} n_dims={nd} n_vec={n_vec} m={m} max_rel_err={rel:.3e} bit_exact_host={exact} bit_identical_rerun={rerun} ik_rel={ik_rel:.3e} {}",
                    verdict(pass)
                );
                if !pass {
                    ok = false;
                }
            }
        }
    }

    // ============================================================ swiglu
    // No isolated oracle row exists: the reference fuses the combine with the
    // gate/up matmuls, so the dump holds only the fused output. Synthetic
    // activations at the three widths of the model (read from the file).
    {
        let sites = [
            ("shexp", "blk.1.ffn_gate_shexp.weight", 2816usize),
            ("dense0", "blk.0.ffn_gate.weight", 10944usize),
            ("expert", "blk.1.ffn_gate_exps.weight", 1408usize),
        ];
        let mut case = 0u32;
        for (site, tensor, ff) in sites {
            let (info, _) = tensor_bytes(&gguf, tensor)?;
            assert_eq!(info.dims[1] as usize, ff, "{tensor} intermediate width");
            for mm in [1usize, 8] {
                case += 1;
                let gate = activations(ff, mm, 7000 + case * 31);
                let up = activations(ff, mm, 9000 + case * 31);
                let g_dev = DeviceBuffer::from_host(stream, &gate)?;
                let u_dev = DeviceBuffer::from_host(stream, &up)?;
                let n = ff * mm;
                let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, n)?;
                elem.enqueue_swiglu(stream, &g_dev, &u_dev, n, &mut y_dev)?;
                stream.synchronize()?;
                let y = y_dev.to_host_vec(stream)?;
                elem.enqueue_swiglu(stream, &g_dev, &u_dev, n, &mut y_dev)?;
                stream.synchronize()?;
                let rerun = bits_equal(&y, &y_dev.to_host_vec(stream)?);

                let y_ref: Vec<f32> = gate
                    .iter()
                    .zip(&up)
                    .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
                    .collect();
                let rel = max_rel_err(&y, &y_ref)?;
                let pass = rel <= PLAIN_BAND && rerun;
                println!(
                    "shape op=swiglu site={site} n={ff} m={mm} max_rel_err={rel:.3e} bit_identical_rerun={rerun} ik_rel=n/a (fused into the matmul in the reference) {}",
                    verdict(pass)
                );
                if !pass {
                    ok = false;
                }
            }
        }
    }

    // =============================================================== add
    // (out, a, b) with the operand pairs proven by element sums before use.
    {
        let chains = [
            ("ffn_inp-1", "kqv_out-1", "l_out-0"),
            ("ffn_out-1", "ffn_moe_out-1", "ffn_shexp-1"),
            ("l_out-1", "ffn_out-1", "ffn_inp-1"),
            ("l_out-0", "ffn_out-0", "ffn_inp-0"),
        ];
        for (out_name, a_name, b_name) in chains {
            let (a_row, a) = load_ref(&man, a_name, 0)?;
            let (b_row, b) = load_ref(&man, b_name, 0)?;
            let (out_row, out) = load_ref(&man, out_name, 0)?;
            expect(&a_row, a_name, "f32", [2048, 6, 1, 1], "in")?;
            expect(&b_row, b_name, "f32", [2048, 6, 1, 1], "in")?;
            expect(&out_row, out_name, "f32", [2048, 6, 1, 1], "ADD")?;
            let (sa, sb, so) = (f64_sum(&a), f64_sum(&b), f64_sum(&out));
            if (sa + sb - so).abs() > 1e-3 * so.abs().max(1.0) {
                println!(
                    "skip op=add out={out_name} reason=\"sum({a_name})+sum({b_name})={} != sum({out_name})={} — dump does not pair these operands\"",
                    sa + sb,
                    so
                );
                continue;
            }
            let n = a.len();
            let a_dev = DeviceBuffer::from_host(stream, &a)?;
            let b_dev = DeviceBuffer::from_host(stream, &b)?;
            let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, n)?;
            elem.enqueue_add(stream, &a_dev, &b_dev, n, &mut y_dev)?;
            stream.synchronize()?;
            let y = y_dev.to_host_vec(stream)?;
            elem.enqueue_add(stream, &a_dev, &b_dev, n, &mut y_dev)?;
            stream.synchronize()?;
            let rerun = bits_equal(&y, &y_dev.to_host_vec(stream)?);

            // Exact: the host reference is the same single add per element.
            let y_ref: Vec<f32> = a.iter().zip(&b).map(|(&av, &bv)| av + bv).collect();
            let rel = max_rel_err(&y, &y_ref)?;
            let exact = bits_equal(&y, &y_ref);
            let ik_rel = max_rel_err(&y, &out)?;
            let pass = exact && rel == 0.0 && rerun;
            println!(
                "shape op=add site={out_name}={a_name}+{b_name} n=2048 m=6 max_rel_err={rel:.3e} bit_exact_host={exact} bit_identical_rerun={rerun} ik_rel={ik_rel:.3e} {}",
                verdict(pass)
            );
            if !pass {
                ok = false;
            }
        }
    }

    // ====================================================== weighted_sum
    // ffn_moe_down-1 [2048, 6, 6] (MUL_MAT_ID) with ffn_moe_weights-1
    // [1, 6, 6] (GET_ROWS) -> ffn_moe_out-1 (MUL_MULTI_ADD); token 0 alone
    // for the m=1 shape.
    {
        let (down_row, down) = load_ref(&man, "ffn_moe_down-1", 0)?;
        let (w_row, wts) = load_ref(&man, "ffn_moe_weights-1", 0)?;
        let (out_row, out) = load_ref(&man, "ffn_moe_out-1", 0)?;
        expect(
            &down_row,
            "ffn_moe_down-1",
            "f32",
            [2048, 6, 6, 1],
            "MUL_MAT_ID",
        )?;
        expect(&w_row, "ffn_moe_weights-1", "f32", [1, 6, 6, 1], "GET_ROWS")?;
        expect(
            &out_row,
            "ffn_moe_out-1",
            "f32",
            [2048, 6, 1, 1],
            "MUL_MULTI_ADD",
        )?;
        let (rows, n_exp) = (2048usize, 6u32);
        for mm in [6usize, 1] {
            let span = n_exp as usize * rows * mm;
            let down_dev = DeviceBuffer::from_host(stream, &down[..span])?;
            let w_dev = DeviceBuffer::from_host(stream, &wts[..n_exp as usize * mm])?;
            let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, rows * mm)?;
            elem.enqueue_weighted_sum(stream, &down_dev, &w_dev, rows, n_exp, mm, &mut y_dev)?;
            stream.synchronize()?;
            let y = y_dev.to_host_vec(stream)?;
            elem.enqueue_weighted_sum(stream, &down_dev, &w_dev, rows, n_exp, mm, &mut y_dev)?;
            stream.synchronize()?;
            let rerun = bits_equal(&y, &y_dev.to_host_vec(stream)?);

            // f64 reference, experts ascending in the buffer's expert axis
            // (`moe`'s scatter order: plain multiply then add per term).
            let mut y_ref = vec![0.0f32; rows * mm];
            for t in 0..mm {
                for d in 0..rows {
                    let mut acc = 0.0f64;
                    for e in 0..n_exp as usize {
                        acc += f64::from(wts[t * n_exp as usize + e])
                            * f64::from(down[(t * n_exp as usize + e) * rows + d]);
                    }
                    y_ref[t * rows + d] = acc as f32;
                }
            }
            let rel = max_rel_err(&y, &y_ref)?;
            let ik_rel = max_rel_err(&y, &out[..rows * mm])?;
            let pass = rel <= REDUCE_BAND && rerun;
            println!(
                "shape op=weighted_sum site=ffn_moe_out-1 rows={rows} n_exp={n_exp} m={mm} max_rel_err={rel:.3e} bit_identical_rerun={rerun} ik_rel={ik_rel:.3e} {}",
                verdict(pass)
            );
            if !pass {
                ok = false;
            }
        }
    }

    // ============================================================ argmax
    {
        let (out_row, logits) = load_ref(&man, "result_output", 0)?;
        expect(
            &out_row,
            "result_output",
            "f32",
            [102400, 1, 1, 1],
            "MUL_MAT",
        )?;
        let host = argmax_ref(&logits);
        let x_dev = DeviceBuffer::from_host(stream, &logits)?;
        let mut idx_dev = DeviceBuffer::<u32>::zeroed(stream, 1)?;
        elem.enqueue_argmax(stream, &x_dev, logits.len(), &mut idx_dev)?;
        stream.synchronize()?;
        let dev = idx_dev.to_host_vec(stream)?[0];
        elem.enqueue_argmax(stream, &x_dev, logits.len(), &mut idx_dev)?;
        stream.synchronize()?;
        let rerun = idx_dev.to_host_vec(stream)?[0] == dev;

        let pass = dev == host && dev == IK_ARGMAX && rerun;
        println!(
            "shape op=argmax n={} argmax_dev={dev} argmax_host={host} bit_identical_rerun={rerun} ik_rel=n/a (integer index; ik_argmax={IK_ARGMAX}) {}",
            logits.len(),
            verdict(pass)
        );
        if !pass {
            ok = false;
        }

        // Tie rule: equal maxima, the lower index must win on both sides. One
        // pair shares a lane (100 and 132 are 32 apart — the lane's strided
        // scan visits both), one pair straddles lanes; the shared-lane pair
        // is what catches a strict-`>`-weakened lane scan, the straddling
        // pair the butterfly's merge order.
        let mut tie = activations(102400, 1, 5555);
        for v in tie.iter_mut() {
            *v *= 0.25;
        }
        tie[100] = 4.0;
        tie[132] = 4.0;
        tie[70000] = 4.0;
        let want = argmax_ref(&tie);
        let t_dev = DeviceBuffer::from_host(stream, &tie)?;
        let mut t_idx = DeviceBuffer::<u32>::zeroed(stream, 1)?;
        elem.enqueue_argmax(stream, &t_dev, tie.len(), &mut t_idx)?;
        stream.synchronize()?;
        let got = t_idx.to_host_vec(stream)?[0];
        let tie_pass = got == want && want == 100;
        println!(
            "shape op=argmax tie_probe n=102400 tied_at=100,132,70000 argmax_dev={got} argmax_host={want} {}",
            verdict(tie_pass)
        );
        if !tie_pass {
            ok = false;
        }

        // ======================================= captured graph (3 ops)
        // embed -> rms_norm -> argmax over resident buffers, eager vs one
        // captured replay: byte identity, three nodes.
        let gain0 = f32_tensor(&gguf, "blk.0.attn_norm.weight", 2048)?;
        let gain0_dev = DeviceBuffer::from_host(stream, &gain0)?;
        let mut emb_y = DeviceBuffer::<f32>::zeroed(stream, m * 2048)?;
        let mut norm_y = DeviceBuffer::<f32>::zeroed(stream, m * 2048)?;
        let mut am_y = DeviceBuffer::<u32>::zeroed(stream, 1)?;
        let run3 = |emb_y: &mut DeviceBuffer<f32>,
                    norm_y: &mut DeviceBuffer<f32>,
                    am_y: &mut DeviceBuffer<u32>|
         -> Result<(), Box<dyn std::error::Error>> {
            elem.enqueue_embed_rows(stream, &emb_dev, &ids_dev, emb_y)?;
            elem.enqueue_rms_norm(stream, emb_y, &gain0_dev, eps, 2048, m, norm_y)?;
            elem.enqueue_argmax(stream, &x_dev, logits.len(), am_y)
        };
        run3(&mut emb_y, &mut norm_y, &mut am_y)?;
        stream.synchronize()?;
        let (e_e, n_e, a_e) = (
            emb_y.to_host_vec(stream)?,
            norm_y.to_host_vec(stream)?,
            am_y.to_host_vec(stream)?,
        );
        let graph = gpu.capture(|_| run3(&mut emb_y, &mut norm_y, &mut am_y))?;
        graph.launch(stream)?;
        stream.synchronize()?;
        let identical = bits_equal(&e_e, &emb_y.to_host_vec(stream)?)
            && bits_equal(&n_e, &norm_y.to_host_vec(stream)?)
            && a_e == am_y.to_host_vec(stream)?;
        let nodes = graph.node_count();
        let g_pass = identical && nodes == 3;
        println!(
            "graph ops=embed+rms_norm+argmax eager_vs_graph_bit_identical={identical} graph_nodes={nodes} {}",
            verdict(g_pass)
        );
        if !g_pass {
            ok = false;
        }
    }

    if !ok {
        eprintln!("FAILED: gate_p4");
        std::process::exit(1);
    }
    println!(
        "PASSED: elem kernels within their bands of the host references; bit-identical reruns; \
         eager == graph replay; chains proven from the dump"
    );
    Ok(())
}

/// One rms_norm case: run the kernel twice on `x`, assert against the host
/// reference and the bit-identical rerun, print the shape line. Returns
/// (pass, printed line).
#[cfg(feature = "gpu")]
#[allow(clippy::type_complexity)]
fn rms_case(
    elem: &bloomery_gpu::elem::ElemKernels,
    stream: &cuda_core::CudaStream,
    x: &[f32],
    k: usize,
    mi: usize,
    gguf: &gguf::Gguf,
    gain_name: &str,
    out_row: &RefRow,
    eps: f32,
) -> Result<(bool, String), Box<dyn std::error::Error>> {
    let gain = f32_tensor(gguf, gain_name, k)?;
    let gain_dev = DeviceBuffer::from_host(stream, &gain)?;
    let x_dev = DeviceBuffer::from_host(stream, x)?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, x.len())?;
    elem.enqueue_rms_norm(stream, &x_dev, &gain_dev, eps, k, mi, &mut y_dev)?;
    stream.synchronize()?;
    let y = y_dev.to_host_vec(stream)?;
    elem.enqueue_rms_norm(stream, &x_dev, &gain_dev, eps, k, mi, &mut y_dev)?;
    stream.synchronize()?;
    let rerun = bits_equal(&y, &y_dev.to_host_vec(stream)?);

    let y_ref = rms_ref(x, k, &gain, eps);
    let rel = max_rel_err(&y, &y_ref)?;
    let ik = ref_tensor_of(out_row)?;
    let ik_rel = max_rel_err(&y, &ik)?;
    let pass = rel <= 1e-5 && rerun;
    Ok((
        pass,
        format!(
            "max_rel_err={rel:.3e} bit_identical_rerun={rerun} ik_rel={ik_rel:.3e} {}",
            verdict(pass)
        ),
    ))
}

/// Load `(name, occurrence)` from the dump with its manifest row.
#[cfg(feature = "gpu")]
fn load_ref(
    man: &[RefRow],
    name: &str,
    occ: u32,
) -> Result<(RefRow, Vec<f32>), Box<dyn std::error::Error>> {
    let row = find_ref_row(man, name, occ)?;
    Ok((row.clone(), ref_tensor_of(row)?))
}

/// Prove a dump row's type, dims and op — the chain check every consumer
/// runs before trusting a tensor (`op` = what produced it). `op = "in"` only
/// checks type and dims (an input's op varies).
#[cfg(feature = "gpu")]
fn expect(
    row: &RefRow,
    what: &str,
    ty: &str,
    ne: [u64; 4],
    op: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if row.ty != ty || row.ne != ne || (op != "in" && row.op != op) {
        return Err(format!(
            "gate_p4: {what}: {} is {} {:?} op {}, want {} {:?} op {}",
            row.name, row.ty, row.ne, row.op, ty, ne, op
        )
        .into());
    }
    Ok(())
}

/// Element sum in f64 — the operand-pair proof of the add chains.
#[cfg(feature = "gpu")]
fn f64_sum(x: &[f32]) -> f64 {
    x.iter().map(|&v| f64::from(v)).sum()
}

/// Host rms_norm reference, the CPU engine's op order (`ops::rms_norm`):
/// f32 squares summed in f64 serially, the mean narrowed to f32,
/// `scale = 1/sqrt(mean + eps)`, then `(scale · gain) · x`.
#[cfg(feature = "gpu")]
fn rms_ref(x: &[f32], k: usize, gain: &[f32], eps: f32) -> Vec<f32> {
    let mm = x.len() / k;
    let mut out = vec![0.0f32; x.len()];
    for t in 0..mm {
        let src = &x[t * k..(t + 1) * k];
        let sum: f64 = src.iter().map(|&v| f64::from(v * v)).sum();
        let mean = (sum / k as f64) as f32;
        let scale = 1.0f32 / (mean + eps).sqrt();
        for i in 0..k {
            out[t * k + i] = (scale * gain[i]) * src[i];
        }
    }
    out
}

/// Host rope reference, the CPU apply fn's op order (`attn::rope_pair`):
/// adjacent pairs (2i, 2i+1), `y0 = x0·cos − x1·sin`, `y1 = x0·sin + x1·cos`.
#[cfg(feature = "gpu")]
fn rope_ref(src: &[f32], cs: &[f32], nd: usize, n_vec: usize, m: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; src.len()];
    for c in 0..m * n_vec {
        let t = c / n_vec;
        let mut i = 0;
        while i < nd {
            let (x0, x1) = (src[c * nd + i], src[c * nd + i + 1]);
            let (cc, s) = (cs[t * nd + i], cs[t * nd + i + 1]);
            out[c * nd + i] = x0 * cc - x1 * s;
            out[c * nd + i + 1] = x0 * s + x1 * cc;
            i += 2;
        }
    }
    out
}

/// Host argmax, the greedy sampler's rule (`forward::argmax`): strict `>`,
/// so ties keep the lower index.
#[cfg(feature = "gpu")]
fn argmax_ref(x: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(feature = "gpu")]
fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[cfg(feature = "gpu")]
fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}
