//! GPU gate for BF16 weights read in their words (`DevWeight::Bf16`, two
//! values a u32 word, the placed load's `CardFormat::Bf16Raw` under
//! `ModelTensors::bf16_words`): the dense site, the head, the head's row
//! list and the embedding lookup, on synthetic tensors written to a GGUF
//! and loaded through the engine's placed path (`ModelTensors::card_format`,
//! `placement::whole_on_card`, `Weights::load_rows`).
//!
//! The tensors: every value a BF16 of a random sign, exponent in 2^-20 to
//! 2^2 and significand, so every product with an activation (uniform on ±2)
//! is a normal f32. Every output is checked, over a NaN sentinel the launch
//! must overwrite:
//! - `written`: finite;
//! - `band`: within the derived band of the reference `Σ_j w_j x_j` in f64,
//!   `w` the host's widened values (`gguf::quant::dequant_row`);
//! - `bits`: equal bit for bit to the F32 rule `runtime::hc_gated::card::f32_dot`
//!   over the widened values (lane `L` the values `32i + L` in increasing
//!   `i`, one fused multiply-add each, then the warp butterfly).
//!
//! 1. site: projection shapes (8,192 × 2,560, 2,560 × 4,096 and
//!    1,024 × 128), plus a 19 × 128 tail, through `site::gemv` at every
//!    width in 1..=8 (the token-major gemv), `site::resident_gemv` at one
//!    (the call GLM's `body::gemv` is) and `site::gemm` at 64 columns (the tile).
//!    On a compute capability 12 device that `site::gemm` runs the
//!    BF16-operand MMA product: its outputs are held to the operand band, and
//!    a subnormal first activation in column 1 leaves every other column's
//!    outputs their bits (`mma.neighbours`, `shared/dense_file.rs`); the bit
//!    checks then read the exact tile's outputs.
//! 2. f32: the same tensors loaded as f32 values (`Weights::load_where`,
//!    `CardFormat::Bf16AsF32`, the load of every body that does not set
//!    `bf16_words`) through the F32 site at 1..=8 and 64 columns: bit for bit
//!    the BF16 site's outputs. A BF16 site over that F32 resident is refused
//!    by name. Each small site call captures as one graph node and replays
//!    bit for bit the eager output.
//! 3. join: two BF16 parts joined at load (`Weights::join_rows`) through
//!    `site::resident_gemv`, bit for bit the rule over the parts' rows; the
//!    same parts loaded as f32 values the same. Over each joint the dense
//!    calls by name hold their outputs to one another bit for bit
//!    (`by_name`, `shared/dense_file.rs`): the m-column gemv, the last part's
//!    rows alone through the gemv and the wide GEMM, and rows past the joint
//!    refused by name.
//! 4. head: a BF16 `output.weight` of 4,099 rows through `head::Head` at one
//!    and three rows (`HeadNorm::Mixed`): its logits the rule bit for bit,
//!    each row's token the host argmax; the Q8_0 head of the same shape the
//!    same against the Q8_0 rule (`card::q8_dot`), the shared path.
//! 5. ids: the head's row list (`Q8F32Kernels::enqueue_bf16_gemv_ids`, what
//!    the MTP draft's head runs over the target's BF16 `output`) at one and
//!    three columns: row `i` bit for bit the full head gemv's row `ids[i]`,
//!    over a list with repeats and the last row; an id past the matrix
//!    raises `FaultSite::TokenId` and writes its row NaN, the other rows
//!    their bits.
//! 6. embed: the BF16 token embedding (`Q38Kernels::enqueue_embed_rows`,
//!    `EmbedRows::Bf16`) gathers each id's widened row bit for bit, with each
//!    row's position and live key count; the Q8_0 table the same against
//!    its dequant, the shared path.
//! 7. refusal: the BF16 tile refuses a row of 96 values (not a whole staged
//!    line) by name; the gemv runs it, every output checked as above.
//! 8. shape: the four BF16 entries compile with no local depot.
//!
//! The band. Every term enters its lane by one fused multiply-add of the
//! exact widened value, so a lane's sum has `k / 32` roundings and the
//! butterfly five more: `|ours − ref| ≤ γ(k/32 + 5) · Σ_j |w_j x_j|`, plus
//! the f64 reference's own `k · u64`.

#[path = "shared/dense_file.rs"]
mod dense_file;

fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_bf16", gate::run())
}

mod gate {
    use crate::dense_file::{
        Decl, by_name, check, columns, decl, host_rows, mma_neighbours, q8_0_bytes,
    };
    use bloomery_gpu::gemm::{Gemm32Kernels, GemmInput, GemmKernels, GemmRoute};
    use bloomery_gpu::head::{Head, HeadNorm};
    use bloomery_gpu::q8f32::Bf16GemvIdsArgs;
    use bloomery_gpu::q38::{EmbedArgs, EmbedRows, Q38Kernels};
    use bloomery_gpu::site::{self, SiteTy, SiteWeight, WideIn, WideKernels};
    use bloomery_gpu::weights::Weights;
    use bloomery_gpu::{Fault, FaultSite, Gpu, LAYER_NONE};
    use bloomery_gpu_gates::rounding::check_mma;
    use bloomery_gpu_gates::{
        GateError, GateRng, checks_failed, no_local_depot, verdict, write_gguf,
    };
    use cuda_core::DeviceBuffer;
    use gguf::Split;
    use gguf::quant::GgmlType;
    use model::placement::{self, ModelTensor, ModelTensors, Role};
    use runtime::hc_gated::card::{f32_dot, q8_dot};
    use std::path::Path;

    const WHAT: &str = "gate_bf16";

    struct Shape {
        name: &'static str,
        rows: usize,
        k: usize,
        seed: u64,
    }

    const SHAPES: [Shape; 4] = [
        Shape {
            name: "blk.3.attn_q.weight",
            rows: 8192,
            k: 2560,
            seed: 0x5eed_b001,
        },
        Shape {
            name: "blk.3.attn_output.weight",
            rows: 2560,
            k: 4096,
            seed: 0x5eed_b002,
        },
        Shape {
            name: "blk.3.ssm_beta.weight",
            rows: 1024,
            k: 128,
            seed: 0x5eed_b003,
        },
        Shape {
            name: "blk.3.tail.weight",
            rows: 19,
            k: 128,
            seed: 0x5eed_b004,
        },
    ];

    /// Columns of the wide tile.
    const WIDE: usize = 64;
    /// The head's and the embedding's rows and width: a row count off the
    /// gemv's eight rows a block.
    const VOCAB: usize = 4099;
    const HIDDEN: usize = 2560;
    const HEAD: &str = bloomery_gpu::weights::HEAD_TENSOR;
    const EMBD: &str = "token_embd.weight";
    const JOIN: [(&str, usize); 2] = [("blk.3.join_a.weight", 96), ("blk.3.join_b.weight", 32)];
    const K96: &str = "blk.3.k96.weight";

    /// `n` BF16 values from `seed` as file bytes (little-endian halves).
    fn bf16_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut rng = GateRng(seed);
        (0..n)
            .flat_map(|_| {
                let sign = (rng.below(2) as u16) << 15;
                let exp = (127 - 20 + rng.below(23)) as u16;
                let bits = sign | exp << 7 | rng.below(128) as u16;
                bits.to_le_bytes()
            })
            .collect()
    }

    /// The role a tensor of this gate's files plays.
    fn role(name: &str) -> Role {
        match name {
            HEAD => Role::Head,
            EMBD => Role::TokenEmbedding,
            _ => Role::Attention,
        }
    }

    /// The tensors of `split` that `keep` names, each whole on card 0 in the
    /// format the placed load of a model whose `bf16_words` names every
    /// tensor gives it: a BF16 matrix in its words.
    fn load_placed(
        gpu: &Gpu,
        split: &Split,
        keep: impl Fn(&str) -> bool,
    ) -> Result<Weights, GateError> {
        let model = ModelTensors {
            tensors: split
                .iter_tensors()
                .map(|(_, t)| ModelTensor {
                    name: t.name.clone(),
                    shard: 0,
                    layer: t.name.starts_with("blk.").then_some(3),
                    role: role(&t.name),
                    ty: t.ty,
                    dims: t.dims.clone(),
                    file_bytes: t.nbytes,
                    gathered_rows: None,
                })
                .collect(),
            layers: 4,
            experts: 0,
            experts_used: 0,
            bf16_words: |_| true,
        };
        let rows = model
            .tensors
            .iter()
            .enumerate()
            .filter(|(_, t)| keep(&t.name))
            .map(|(i, t)| {
                placement::whole_on_card(i, t, 0, &model).map(|mut row| {
                    // This gate retains the BF16 head pins; gate_mxfp8 checks default conversion.
                    if t.name == HEAD && t.ty == GgmlType::BF16 {
                        let format = placement::CardFormat::Bf16Raw;
                        row.segments[0].format = placement::Format::Card(format);
                        row.segments[0].resident_bytes = format
                            .resident_bytes(t.ty, t.dims[0], t.dims[1])
                            .expect("fixed BF16 fixture dimensions");
                    }
                    row
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("rows: {e}"))?;
        Ok(Weights::load_rows(gpu.stream(), split, &model, &rows, 0)?)
    }

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let gpu = Gpu::new()?;
        println!("bf16 device={}", gpu.device_name()?);
        let dir = std::env::temp_dir().join(format!("bloomery-bf16-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let outcome = cases(&gpu, &dir);
        let cleanup = std::fs::remove_dir_all(&dir).map_err(|e| format!("cleanup: {e}"));
        let mut ok = outcome?;
        cleanup?;
        ok &= no_local_depot(&[
            "bf16_gemv",
            "bf16_gemv_ids",
            "bf16_tile_gemm",
            "embed_rows_bf16",
        ])?;
        if ok { Ok(()) } else { Err(checks_failed()) }
    }

    fn cases(gpu: &Gpu, dir: &Path) -> Result<bool, GateError> {
        let mut decls: Vec<Decl> = SHAPES
            .iter()
            .map(|s| {
                let bytes = bf16_bytes(s.rows * s.k, s.seed);
                decl(s.name, GgmlType::BF16, (s.rows, s.k), bytes)
            })
            .collect();
        for (i, (name, rows)) in JOIN.iter().enumerate() {
            let bytes = bf16_bytes(rows * 256, 0x5eed_b010 + i as u64);
            decls.push(decl(name, GgmlType::BF16, (*rows, 256), bytes));
        }
        decls.push(decl(
            K96,
            GgmlType::BF16,
            (8, 96),
            bf16_bytes(8 * 96, 0x5eed_b020),
        ));
        let head = bf16_bytes(VOCAB * HIDDEN, 0x5eed_b030);
        decls.push(decl(HEAD, GgmlType::BF16, (VOCAB, HIDDEN), head));
        let embd = bf16_bytes(VOCAB * HIDDEN, 0x5eed_b031);
        decls.push(decl(EMBD, GgmlType::BF16, (VOCAB, HIDDEN), embd));
        let bf16_path = dir.join("bf16.gguf");
        write_gguf(&bf16_path, &[], &decls)?;
        let q8_decls = [
            decl(
                HEAD,
                GgmlType::Q8_0,
                (VOCAB, HIDDEN),
                q8_0_bytes(VOCAB, HIDDEN, 0x5eed_b040),
            ),
            decl(
                EMBD,
                GgmlType::Q8_0,
                (VOCAB, HIDDEN),
                q8_0_bytes(VOCAB, HIDDEN, 0x5eed_b041),
            ),
        ];
        let q8_path = dir.join("q8_0.gguf");
        write_gguf(&q8_path, &[], &q8_decls)?;
        let open = |p: &Path| Split::open(p).map_err(|e| format!("open {}: {e}", p.display()));
        let (split, q8_split) = (open(&bf16_path)?, open(&q8_path)?);
        let w = load_placed(gpu, &split, |_| true)?;
        let q8_w = load_placed(gpu, &q8_split, |_| true)?;

        let g32 = Gemm32Kernels::load(gpu.context())?;
        let gemm = GemmKernels::load(gpu.context())?;
        let mut route = GemmRoute::new(gpu.stream(), WIDE, 1)?;
        gemm.enqueue_route_dense(gpu.stream(), WIDE, &mut route, gpu.unlabelled_sink())?;
        let wide = (&g32, &gemm, &route);
        let host = |d: &Decl| host_rows(d.2, &d.3, d.1[1] as usize, d.1[0] as usize);

        let mut ok = true;
        for (s, d) in SHAPES.iter().zip(&decls) {
            ok &= site_case(gpu, wide, (&split, &w), s, &host(d))?;
        }
        let parts: Vec<Vec<f32>> = decls[SHAPES.len()..][..JOIN.len()]
            .iter()
            .flat_map(host)
            .collect();
        let joined = load_placed(gpu, &split, |n| JOIN.iter().any(|j| j.0 == n))?;
        let widened = Weights::load_where(gpu.stream(), &split, |n| JOIN.iter().any(|j| j.0 == n))?;
        let wide_at = (&g32, &gemm, &route, WIDE);
        ok &= join_case(gpu, wide_at, ("bf16", joined), &parts)?;
        ok &= join_case(gpu, wide_at, ("f32", widened), &parts)?;
        ok &= k96_case(gpu, &w, &host(&decls[SHAPES.len() + JOIN.len()]))?;
        let head_rows = host(&decls[SHAPES.len() + JOIN.len() + 1]);
        let embd_rows = host(&decls[SHAPES.len() + JOIN.len() + 2]);
        let q8_head = host(&q8_decls[0]);
        let q8_embd = host(&q8_decls[1]);
        ok &= head_case(gpu, &w, "bf16", &head_rows, f32_dot)?;
        ok &= head_case(gpu, &q8_w, "q8_0", &q8_head, q8_dot)?;
        ok &= ids_case(gpu, &w, &head_rows)?;
        let q38 = Q38Kernels::load(gpu.context())?;
        ok &= embed_case(gpu, &q38, &w, "bf16", &embd_rows)?;
        ok &= embed_case(gpu, &q38, &q8_w, "q8_0", &q8_embd)?;
        Ok(ok)
    }

    /// Clauses 1 and 2 at one shape (module doc).
    fn site_case(
        gpu: &Gpu,
        (g32, gemm, route): (&Gemm32Kernels, &GemmKernels, &GemmRoute),
        (split, w): (&Split, &Weights),
        s: &Shape,
        wf: &[Vec<f32>],
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let (name, rows, k) = (s.name, s.rows, s.k);
        let ty = site::file_site(split, WHAT, name, (rows, k), &[SiteTy::Bf16])?;
        site::site(w, WHAT, name, (rows, k), ty)?;
        let f32_w = Weights::load_where(stream, split, |n| n == name)?;
        let token_major = |at: usize| (at % rows, at / rows);
        let mut ok = true;
        // m = 0 marks the resident gemv's one column.
        for (i, m) in (0..=8).chain([WIDE]).enumerate() {
            let cols = m.max(1);
            let seed = s.seed ^ (0x1000 + i as u64);
            let (x, x_dev, mut y_dev) = columns(gpu, (cols, k, rows), seed, 1.0)?;
            let call = match m {
                0 => {
                    site::resident_gemv(gpu, w, name, &x_dev, &mut y_dev)?;
                    "resident_gemv"
                }
                WIDE => {
                    let x_in = WideIn {
                        q128: None,
                        q32: None,
                        f32: Some(&x_dev),
                    };
                    let kernels = WideKernels {
                        gemm,
                        g32: Some(g32),
                    };
                    let src = (ty, w, SiteWeight::Resident(name));
                    let table = (route, GemmInput::PerSlot);
                    site::gemm(gpu, &kernels, src, rows, &x_in, table, m, &mut y_dev)?;
                    if bloomery_gpu::dense_mma(gpu.context())? {
                        stream.synchronize()?;
                        let mma = y_dev.to_host_vec(stream)?;
                        ok &= check_mma(name, (&mma, &x, k), wf);
                        let tag = format!("shape={name} rows={rows} k={k} m={m}");
                        ok &= mma_neighbours(
                            gpu,
                            &kernels,
                            (src, route),
                            (&x, k, rows),
                            ("bf16", &tag),
                        )?;
                        // PIN(2026-10-22): the BF16 operand rounding of a compute capability 12
                        // device replaces the site's F32 bit contract with the band/KL check above;
                        // the exact tile's output replaces the site's here, so the bit checks below
                        // hold for the exact tile only.
                        gpu.q8f32().enqueue_bf16_tile(
                            stream,
                            w.bf16(WHAT, name)?,
                            &x_dev,
                            m,
                            &mut y_dev,
                        )?;
                    }
                    "gemm"
                }
                _ => {
                    site::gemv(gpu, g32, (ty, w, name), &x_dev, m, &mut y_dev)?;
                    "gemv"
                }
            };
            stream.synchronize()?;
            let y = y_dev.to_host_vec(stream)?;
            let rule = |r: usize, c: usize| f32_dot(&wf[r], &x[c * k..][..k]);
            let tag = format!("shape={name} rows={rows} k={k} call={call} m={cols}");
            ok &= check(("bf16", "site"), &tag, (&y, &x, k), wf, &token_major, &rule);
            {
                // The F32 site over the values loaded as f32: the same bits.
                let mut f_dev = DeviceBuffer::from_host(stream, &vec![f32::NAN; cols * rows])?;
                site::gemv(
                    gpu,
                    g32,
                    (SiteTy::F32, &f32_w, name),
                    &x_dev,
                    cols,
                    &mut f_dev,
                )?;
                stream.synchronize()?;
                let f = f_dev.to_host_vec(stream)?;
                let same = f.iter().zip(&y).all(|(a, b)| a.to_bits() == b.to_bits());
                println!(
                    "bf16 clause=f32.bits {tag} the f32 site's outputs equal={same} {}",
                    verdict(same)
                );
                ok &= same;
                if (1..=8).contains(&m) {
                    // Poisoned again, so a tile that skips a write cannot pass on the site's values.
                    f_dev = DeviceBuffer::from_host(stream, &vec![f32::NAN; cols * rows])?;
                    gpu.q8f32().enqueue_bf16_tile(
                        stream,
                        w.bf16(WHAT, name)?,
                        &x_dev,
                        m,
                        &mut f_dev,
                    )?;
                    stream.synchronize()?;
                    let tile = f_dev.to_host_vec(stream)?;
                    let same = bloomery_gpu_gates::bits_equal(&tile, &y);
                    println!("bf16 clause=tile.bits {tag} equal={same} {}", verdict(same));
                    ok &= same;
                }
            }
            if (1..=8).contains(&m) {
                y_dev.zero_async(stream)?;
                stream.synchronize()?;
                let graph =
                    gpu.capture(|_| site::gemv(gpu, g32, (ty, w, name), &x_dev, m, &mut y_dev))?;
                graph.launch(stream)?;
                stream.synchronize()?;
                let replay = y_dev.to_host_vec(stream)?;
                let same = bloomery_gpu_gates::bits_equal(&replay, &y);
                let pass = same && graph.node_count() == 1;
                println!(
                    "bf16 clause=graph.bits {tag} equal={same} nodes={} {}",
                    graph.node_count(),
                    verdict(pass)
                );
                ok &= pass;
            }
        }
        let got = site::site(&f32_w, WHAT, name, (rows, k), ty);
        let refused = got
            .as_ref()
            .is_err_and(|e| e.to_string().contains(name) && e.to_string().contains("as F32"));
        println!(
            "bf16 clause=f32.refusal tensor={name} a bf16 site over the f32 resident: {} {}",
            match &got {
                Ok(()) => "accepted".to_owned(),
                Err(e) => e.to_string(),
            },
            verdict(refused)
        );
        Ok(ok && refused)
    }

    /// Clause 3 (module doc), over the parts resident as `tag`.
    fn join_case(
        gpu: &Gpu,
        wide: (&Gemm32Kernels, &GemmKernels, &GemmRoute, usize),
        (tag, mut w): (&str, Weights),
        wf: &[Vec<f32>],
    ) -> Result<bool, GateError> {
        const JOINT: &str = "derived.bf16_joint";
        let stream = gpu.stream();
        w.join_rows(stream, &JOIN.map(|(n, _)| n), JOINT.to_string())?;
        let (rows, k) = (wf.len(), 256);
        let (x, x_dev, mut y_dev) = columns(gpu, (1, k, rows), 0x5eed_b011, 1.0)?;
        site::resident_gemv(gpu, &w, JOINT, &x_dev, &mut y_dev)?;
        stream.synchronize()?;
        let y = y_dev.to_host_vec(stream)?;
        let rule = |r: usize, c: usize| f32_dot(&wf[r], &x[c * k..][..k]);
        let shown = format!("parts={} rows={rows} k={k} m=1", JOIN.len());
        let joined = check((tag, "join"), &shown, (&y, &x, k), wf, &|at| (at, 0), &rule);
        let part = (JOIN[0].1, JOIN[1].1);
        let named = by_name(gpu, wide, (tag, &w, JOINT), (rows, k), part, 0x5eed_b012)?;
        Ok(joined && named)
    }

    /// Clause 7 (module doc).
    fn k96_case(gpu: &Gpu, w: &Weights, wf: &[Vec<f32>]) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let words = w.bf16(WHAT, K96)?;
        let x_host = [0.5f32; 2 * 96];
        let x = DeviceBuffer::from_host(stream, &x_host)?;
        let mut y = DeviceBuffer::from_host(stream, &[f32::NAN; 2 * 8])?;
        gpu.q8f32()
            .enqueue_bf16_gemv(stream, words, &x, 2, &mut y)?;
        stream.synchronize()?;
        let got = y.to_host_vec(stream)?;
        let rule = |r: usize, c: usize| f32_dot(&wf[r], &x_host[c * 96..][..96]);
        let gemv_ok = check(
            ("bf16", "k96"),
            "rows=8 k=96 m=2",
            (&got, &x_host, 96),
            wf,
            &|at| (at / 2, at % 2),
            &rule,
        );
        for m in 1..=8 {
            let (host, input, mut output) = columns(gpu, (m, 96, 8), 0x5eed_b021, 1.0)?;
            site::resident_gemv_cols(gpu, None, (w, K96), &input, m, &mut output)?;
            stream.synchronize()?;
            let got = output.to_host_vec(stream)?;
            let rule = |r: usize, c: usize| f32_dot(&wf[r], &host[c * 96..][..96]);
            if !check(
                ("bf16", "k96.site"),
                &format!("rows=8 k=96 m={m}"),
                (&got, &host, 96),
                wf,
                &|at| (at % 8, at / 8),
                &rule,
            ) {
                return Err(checks_failed());
            }
        }
        let tile = gpu.q8f32().enqueue_bf16_tile(stream, words, &x, 2, &mut y);
        let mma = gpu.q8f32().enqueue_bf16_mma(stream, words, &x, 2, &mut y);
        stream.synchronize()?;
        let pass = gemv_ok
            && [&tile, &mma]
                .iter()
                .all(|r| r.as_ref().is_err_and(|e| e.to_string().contains("96")));
        println!(
            "bf16 clause=refusal k=96 gemv_outputs={gemv_ok} tile={:?} mma={:?} {}",
            tile.as_ref().map_err(ToString::to_string),
            mma.as_ref().map_err(ToString::to_string),
            verdict(pass)
        );
        Ok(pass)
    }

    /// The host argmax of each of `m` rows of `logits` (`[v·m + c]`): the
    /// first largest.
    fn argmax(logits: &[f32], m: usize) -> Vec<u32> {
        (0..m)
            .map(|c| {
                let mut best = 0;
                for v in 1..logits.len() / m {
                    if logits[v * m + c] > logits[best * m + c] {
                        best = v;
                    }
                }
                best as u32
            })
            .collect()
    }

    /// Clause 4 at one head type (module doc).
    fn head_case(
        gpu: &Gpu,
        w: &Weights,
        ty: &str,
        wf: &[Vec<f32>],
        dot: fn(&[f32], &[f32]) -> f32,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        for m in [1, 3] {
            let mut head = Head::with_norm(gpu, w, 1e-6, m, HeadNorm::Mixed)?;
            let mut rng = GateRng(0x5eed_b050 + m as u64);
            let x: Vec<f32> = (0..m * HIDDEN).map(|_| rng.uni(-2.0, 2.0)).collect();
            head.set_input(gpu, &x)?;
            head.enqueue(gpu, w)?;
            let tokens = head.tokens(gpu)?;
            let logits = head.logits_to_host(gpu)?;
            let rule = |r: usize, c: usize| dot(&wf[r], &x[c * HIDDEN..][..HIDDEN]);
            let tag = format!("type={ty} rows={VOCAB} k={HIDDEN} m={m}");
            let row_major = |at: usize| (at / m, at % m);
            ok &= check(
                ("bf16", "head"),
                &tag,
                (&logits, &x, HIDDEN),
                wf,
                &row_major,
                &rule,
            );
            let want = argmax(&logits, m);
            let pass = tokens == want;
            println!(
                "bf16 clause=head.tokens {tag} tokens={tokens:?} argmax={want:?} {}",
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    /// Clause 5 (module doc).
    fn ids_case(gpu: &Gpu, w: &Weights, wf: &[Vec<f32>]) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let words = w.bf16(WHAT, HEAD)?;
        let mut rng = GateRng(0x5eed_b060);
        let mut ids: Vec<u32> = (0..34).map(|_| rng.below(VOCAB) as u32).collect();
        ids.extend([VOCAB as u32 - 1, ids[3], 0]);
        let ids_dev = DeviceBuffer::from_host(stream, &ids)?;
        let mut ok = true;
        for m in [1, 3] {
            let (x, x_dev, mut full) =
                columns(gpu, (m, HIDDEN, VOCAB), 0x5eed_b061 + m as u64, 1.0)?;
            gpu.q8f32()
                .enqueue_bf16_gemv(stream, words, &x_dev, m, &mut full)?;
            let mut y = DeviceBuffer::from_host(stream, &vec![f32::NAN; ids.len() * m])?;
            let args = Bf16GemvIdsArgs {
                w: words,
                ids: &ids_dev,
                rows: ids.len(),
                x: &x_dev,
                m,
                y: &mut y,
            };
            gpu.q8f32()
                .enqueue_bf16_gemv_ids(stream, args, gpu.unlabelled_sink())?;
            stream.synchronize()?;
            let (full, y) = (full.to_host_vec(stream)?, y.to_host_vec(stream)?);
            let same = (0..ids.len() * m)
                .all(|at| y[at].to_bits() == full[ids[at / m] as usize * m + at % m].to_bits());
            let fault = gpu.take_fault()?;
            let rule = |r: usize, c: usize| f32_dot(&wf[r], &x[c * HIDDEN..][..HIDDEN]);
            let listed = |at: usize| (ids[at / m] as usize, at % m);
            let tag = format!("rows={} k={HIDDEN} m={m}", ids.len());
            ok &= check(("bf16", "ids"), &tag, (&y, &x, HIDDEN), wf, &listed, &rule);
            let pass = same && fault.is_none();
            println!(
                "bf16 clause=ids.full {tag} each row the full head's row ids[i] equal={same} \
                 fault={fault:?} {}",
                verdict(pass)
            );
            ok &= pass;
        }
        // An id past the matrix: its row NaN and a raise, the other row its bits.
        let bad = DeviceBuffer::from_host(stream, &[5u32, VOCAB as u32])?;
        let (_, x_dev, mut full) = columns(gpu, (1, HIDDEN, VOCAB), 0x5eed_b064, 1.0)?;
        gpu.q8f32()
            .enqueue_bf16_gemv(stream, words, &x_dev, 1, &mut full)?;
        let mut y = DeviceBuffer::from_host(stream, &[0.0f32; 2])?;
        let args = Bf16GemvIdsArgs {
            w: words,
            ids: &bad,
            rows: 2,
            x: &x_dev,
            m: 1,
            y: &mut y,
        };
        gpu.q8f32()
            .enqueue_bf16_gemv_ids(stream, args, gpu.unlabelled_sink())?;
        stream.synchronize()?;
        let (full, y) = (full.to_host_vec(stream)?, y.to_host_vec(stream)?);
        let fault = gpu.take_fault()?;
        let want = Some(Fault::of_sites(LAYER_NONE, &[FaultSite::TokenId]));
        let pass = y[0].to_bits() == full[5].to_bits() && y[1].is_nan() && fault == want;
        println!(
            "bf16 clause=ids.past id={VOCAB} rows={y:?} fault={fault:?} want={want:?} {}",
            verdict(pass)
        );
        Ok(ok && pass)
    }

    /// Clause 6 at one table type (module doc).
    fn embed_case(
        gpu: &Gpu,
        q38: &Q38Kernels,
        w: &Weights,
        ty: &str,
        rows: &[Vec<f32>],
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let ids = [0u32, 17, VOCAB as u32 - 1, 5, 17];
        let (pos0, first) = (7u32, 2usize);
        let ids_dev = DeviceBuffer::from_host(stream, &ids)?;
        let pos0_dev = DeviceBuffer::from_host(stream, &[pos0])?;
        let mut y = DeviceBuffer::from_host(stream, &vec![f32::NAN; ids.len() * HIDDEN])?;
        let mut pos = DeviceBuffer::from_host(stream, &[0u32; 5])?;
        let mut n_keys = DeviceBuffer::from_host(stream, &[0u32; 5])?;
        let table = EmbedRows::of(w, WHAT, EMBD)?;
        let is_bf16 = matches!(table, EmbedRows::Bf16(_));
        q38.enqueue_embed_rows(
            stream,
            EmbedArgs {
                table,
                ids: &ids_dev,
                pos0: &pos0_dev,
                first,
                fault: gpu.unlabelled_sink(),
                y: &mut y,
                pos: &mut pos,
                n_keys: &mut n_keys,
            },
        )?;
        stream.synchronize()?;
        let y = y.to_host_vec(stream)?;
        let (pos, n_keys) = (pos.to_host_vec(stream)?, n_keys.to_host_vec(stream)?);
        let fault = gpu.take_fault()?;
        let bits = ids.iter().enumerate().all(|(t, &id)| {
            y[t * HIDDEN..][..HIDDEN]
                .iter()
                .zip(&rows[id as usize])
                .all(|(a, b)| a.to_bits() == b.to_bits())
        });
        let want_pos: Vec<u32> = (0..ids.len() as u32)
            .map(|t| pos0 + first as u32 + t)
            .collect();
        let want_keys: Vec<u32> = want_pos.iter().map(|p| p + 1).collect();
        let resident = is_bf16 == (ty == "bf16");
        let pass = bits && pos == want_pos && n_keys == want_keys && fault.is_none() && resident;
        println!(
            "bf16 clause=embed type={ty} ids={ids:?} rows_bit_equal={bits} pos={pos:?} \
             n_keys={n_keys:?} fault={fault:?} read_as_{ty}={resident} {}",
            verdict(pass)
        );
        Ok(pass)
    }
}
