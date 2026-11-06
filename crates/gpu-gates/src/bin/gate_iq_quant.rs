//! GPU gate for the card's IQ1_S and IQ2_XXS quantizer (`bloomery_gpu::iq_quant`):
//! its bytes are ggml's, the quantizer llama-quantize runs, row for row.
//!
//! The oracle is `iq_quant_ref` (`tools/ref/iq_quant_ref.cpp`, `just
//! build-iq-quant-ref`), linked to the K3.1 quantization tree's libggml-base:
//! it widens the source rows with ggml's `to_float` and quantizes each
//! expert's rows with `ggml_quantize_chunk` and that expert's importance row.
//! The card gets the same source bytes and importance matrix. Every input is
//! fixed (seeded), so the gate is bit identity.
//!
//! 1. mxfp4: MXFP4 rows (the K3.1 experts' file type) of 3584 values, two
//!    experts of 12 rows with their own importance rows, a zero in every 13
//!    (a column the calibration never reached): E8M0 scales over 115..=126
//!    and random codes, so a sub-block holds at most 15 distinct values and
//!    its sort meets ties everywhere, `-0` beside `0`.
//! 2. f32: 24 rows of 2048 values a case — uniform, a sum of four uniforms,
//!    zero sub-blocks between small integers, sub-blocks of 1e-14 (IQ1_S's
//!    zero threshold but not IQ2_XXS's), one large value in eight,
//!    importance with zeros, and values of three decimals (ties) under
//!    importance with zeros: a weight of zero makes IQ1_S's split search
//!    meet equal costs, so its pick depends on the sort's order of equal
//!    values, ggml's stable one.
//! 3. host: `gguf::iq_quant::quantize_row` on the first two rows of every case
//!    is the same bytes, so the host rule is the card's.
//! 4. set: `iq_quant_set` (built beside this binary) over a two-shard set —
//!    two MXFP4 expert stacks of three experts the plan types IQ1_S and
//!    IQ2_XXS, an F32 tensor planned F32 and one planned Q4_K — with an
//!    imatrix whose counts round to 3, to 0 (importance all ones) and 5, at
//!    `--batch 2`: the output's shards hold the input's metadata, names,
//!    dims and order, the copied tensors' bytes, and ggml's bytes for the
//!    stacks under the importance llama-quantize computes. A plan that lacks
//!    a tensor is refused and leaves no file.
//!
//! A row that differs prints its case, type, row and first differing
//! super-block and byte.

#[cfg(feature = "gpu")]
use bloomery_gpu::Gpu;
#[cfg(feature = "gpu")]
use bloomery_gpu::iq_quant::IqQuant;
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::{GateError, data_dir, verdict};
#[cfg(feature = "gpu")]
use cuda_core::DeviceBuffer;
#[cfg(feature = "gpu")]
use gguf::iq_quant::{GridTables, IqKind, quantize_row};
#[cfg(feature = "gpu")]
use gguf::quant::{GgmlType, dequant_row};
#[cfg(feature = "gpu")]
use std::path::{Path, PathBuf};

/// A tensor of the set clause's input: name, dims, type and bytes.
#[cfg(feature = "gpu")]
type SetTensor = (String, Vec<u64>, GgmlType, Vec<u8>);

/// Write one GGUF file of `kvs` and `tensors`.
#[cfg(feature = "gpu")]
fn write_gguf(
    path: &Path,
    kvs: &[(String, gguf::Value)],
    tensors: &[SetTensor],
) -> Result<(), GateError> {
    use gguf::write::{Layout, TensorDecl, Writer};
    let decls = tensors
        .iter()
        .map(|(name, dims, ty, bytes)| TensorDecl {
            name: name.clone(),
            dims: dims.clone(),
            type_id: ty.as_u32(),
            nbytes: bytes.len() as u64,
        })
        .collect();
    let mut w = Writer::new(std::fs::File::create(path)?, Layout::new(kvs, decls)?)?;
    for (name, _, _, bytes) in tensors {
        w.tensor(name, bytes)?;
    }
    w.finish()?;
    Ok(())
}

/// Clause 4: `iq_quant_set` over a two-shard set, against the input and ggml.
#[cfg(feature = "gpu")]
fn set_clause(bin: &Path, dir: &Path, rng: &mut Rng) -> Result<bool, GateError> {
    use gguf::Value;
    let tool = std::env::current_exe()?.with_file_name("iq_quant_set");
    if !tool.is_file() {
        return Err(format!(
            "no {} (the recipe builds it beside the gate)",
            tool.display()
        )
        .into());
    }
    let (ne0, ne1, n_exp) = (512usize, 4usize, 3usize);
    let mxfp4 = |rng: &mut Rng| -> Vec<u8> {
        let mut v = Vec::new();
        for _ in 0..ne0 * ne1 * n_exp / 32 {
            v.push(115 + (rng.next() % 12) as u8);
            v.extend((0..16).map(|_| rng.next() as u8));
        }
        v
    };
    let stack = vec![ne0 as u64, ne1 as u64, n_exp as u64];
    let gate_name = "blk.0.ffn_gate_exps.weight";
    let down_name = "blk.0.ffn_down_exps.weight";
    let f32_bytes = |rng: &mut Rng, n: usize| -> Vec<u8> {
        (0..n)
            .flat_map(|_| (rng.unit() - 0.5).to_le_bytes())
            .collect()
    };
    let shards: [Vec<SetTensor>; 2] = [
        vec![
            (
                "a.f32".into(),
                vec![256, 3],
                GgmlType::F32,
                f32_bytes(rng, 768),
            ),
            (gate_name.into(), stack.clone(), GgmlType::MXFP4, mxfp4(rng)),
        ],
        vec![
            (down_name.into(), stack, GgmlType::MXFP4, mxfp4(rng)),
            (
                "b.q4".into(),
                vec![256, 2],
                GgmlType::F32,
                f32_bytes(rng, 512),
            ),
        ],
    ];
    let src_prefix = dir.join("set-in");
    for (i, tensors) in shards.iter().enumerate() {
        let kvs = vec![
            (
                "general.architecture".to_string(),
                Value::String("iqset".to_string()),
            ),
            ("split.no".to_string(), Value::U16(i as u16)),
            ("split.count".to_string(), Value::U16(2)),
            ("split.tensors.count".to_string(), Value::I32(4)),
        ];
        let path = PathBuf::from(format!(
            "{}-{:05}-of-00002.gguf",
            src_prefix.display(),
            i + 1
        ));
        write_gguf(&path, &kvs, tensors)?;
    }
    // The imatrix: per stack, sums over ne0 x n_exp and counts that round to 3, 0 and 5.
    let counts = [2.6f32, 0.3, 5.0];
    let mut im_tensors: Vec<SetTensor> = Vec::new();
    let mut want_qw: Vec<Vec<f32>> = Vec::new();
    for name in [gate_name, down_name] {
        let sums: Vec<f32> = (0..ne0 * n_exp).map(|_| 10.0 * rng.unit()).collect();
        want_qw.push(
            (0..ne0 * n_exp)
                .map(|j| {
                    let c = counts[j / ne0].round();
                    if c > 0.0 { sums[j] / c } else { 1.0 }
                })
                .collect(),
        );
        im_tensors.push((
            format!("{name}.in_sum2"),
            vec![ne0 as u64, n_exp as u64],
            GgmlType::F32,
            sums.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ));
        im_tensors.push((
            format!("{name}.counts"),
            vec![1, n_exp as u64],
            GgmlType::F32,
            counts.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ));
    }
    let im = dir.join("set.imatrix.gguf");
    write_gguf(&im, &[], &im_tensors)?;
    let plan_rows = [
        ("a.f32", "F32", "256x3"),
        (gate_name, "IQ1_S", "512x4x3"),
        (down_name, "IQ2_XXS", "512x4x3"),
        ("b.q4", "Q4_K", "256x2"),
    ];
    let plan_text = |rows: &[(&str, &str, &str)]| -> String {
        rows.iter()
            .map(|(n, t, d)| format!("{n}\t{t}\t{d}\t0\n"))
            .collect()
    };
    let first = PathBuf::from(format!("{}-00001-of-00002.gguf", src_prefix.display()));
    let out_prefix = dir.join("set-out");
    let run_tool = |plan: &Path| -> Result<std::process::ExitStatus, GateError> {
        Ok(std::process::Command::new(&tool)
            .arg(&first)
            .args(["--imatrix".as_ref(), im.as_os_str()])
            .args(["--plan".as_ref(), plan.as_os_str()])
            .args(["--out".as_ref(), out_prefix.as_os_str()])
            .args(["--batch", "2"])
            .status()?)
    };
    // A plan without b.q4 is refused before any write.
    let short = dir.join("set-short.tsv");
    std::fs::write(&short, plan_text(&plan_rows[..3]))?;
    let refused = !run_tool(&short)?.success()
        && std::fs::read_dir(dir)?
            .filter_map(Result::ok)
            .all(|e| !e.file_name().to_string_lossy().starts_with("set-out"));
    let plan = dir.join("set-plan.tsv");
    std::fs::write(&plan, plan_text(&plan_rows))?;
    if !run_tool(&plan)?.success() {
        println!("FAIL set: iq_quant_set ended non-zero on the whole plan");
        return Ok(false);
    }
    let input = gguf::Split::open(&first)?;
    let output = gguf::Split::open(format!("{}-00001-of-00002.gguf", out_prefix.display()))?;
    let mut ok = refused;
    if !refused {
        println!("  set: the short plan was not refused, or left a file");
    }
    for i in 0..2 {
        let (a, b) = (
            input.shard(i).expect("2 shards"),
            output.shard(i).expect("2 shards"),
        );
        if a.keys() != b.keys() {
            println!("  set: shard {} metadata differs", i + 1);
            ok = false;
        }
        let names = |g: &gguf::Gguf| -> Vec<(String, Vec<u64>)> {
            g.iter_tensors()
                .map(|t| (t.name.clone(), t.dims.clone()))
                .collect()
        };
        if names(a) != names(b) {
            println!("  set: shard {} names or dims differ", i + 1);
            ok = false;
        }
    }
    for (name, kind, qw) in [
        (gate_name, IqKind::Iq1S, &want_qw[0]),
        (down_name, IqKind::Iq2Xxs, &want_qw[1]),
    ] {
        let case = Case {
            name,
            ty: GgmlType::MXFP4,
            src: input.data(name)?.to_vec(),
            n_per_row: ne0,
            rows: ne1 * n_exp,
            rows_per_expert: ne1,
            qw: qw.clone(),
        };
        let want = oracle(bin, dir, &case, kind)?;
        let (_, t) = output.find(name).expect("names checked above");
        let ty_ok = t.ty
            == match kind {
                IqKind::Iq1S => GgmlType::IQ1_S,
                IqKind::Iq2Xxs => GgmlType::IQ2_XXS,
            };
        let same = same_rows("set", kind, ne0, output.data(name)?, &want);
        if !(ty_ok && same) {
            println!("  set: {name} type {:?}, bytes = ggml {same}", t.ty);
            ok = false;
        }
    }
    for name in ["a.f32", "b.q4"] {
        if input.data(name)? != output.data(name)? {
            println!("  set: {name} is not copied byte for byte");
            ok = false;
        }
    }
    println!(
        "{} set: iq_quant_set over two shards: metadata, names, dims and copies equal the input, the stacks equal \
         ggml, the short plan refused with no file {refused}",
        verdict(ok)
    );
    Ok(ok)
}

fn main() -> std::process::ExitCode {
    #[cfg(feature = "gpu")]
    let r = run();
    #[cfg(not(feature = "gpu"))]
    let r: Result<(), Box<dyn std::error::Error>> =
        Err("gate_iq_quant needs the `gpu` feature".into());
    bloomery_gpu_gates::exit_with("gate_iq_quant", r)
}

/// One input: source rows of a type, their geometry and the importance matrix.
#[cfg(feature = "gpu")]
struct Case {
    name: &'static str,
    ty: GgmlType,
    src: Vec<u8>,
    n_per_row: usize,
    rows: usize,
    rows_per_expert: usize,
    qw: Vec<f32>,
}

/// xorshift64, the gate's one generator.
#[cfg(feature = "gpu")]
struct Rng(u64);

#[cfg(feature = "gpu")]
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / 16_777_216.0
    }
}

/// `n` positive importance values, a zero every `zero_every` when nonzero.
#[cfg(feature = "gpu")]
fn importance(rng: &mut Rng, n: usize, zero_every: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            if zero_every != 0 && i % zero_every == 0 {
                0.0
            } else {
                1e-3 + 10.0 * rng.unit()
            }
        })
        .collect()
}

/// The MXFP4 case: two experts of 12 rows of 3584 values.
#[cfg(feature = "gpu")]
fn mxfp4_case(rng: &mut Rng) -> Case {
    let (n, per, experts) = (3584, 12, 2);
    let rows = per * experts;
    let mut src = Vec::with_capacity(rows * n / 32 * 17);
    for _ in 0..rows * n / 32 {
        src.push(115 + (rng.next() % 12) as u8);
        for _ in 0..16 {
            src.push(rng.next() as u8);
        }
    }
    Case {
        name: "mxfp4",
        ty: GgmlType::MXFP4,
        src,
        n_per_row: n,
        rows,
        rows_per_expert: per,
        qw: importance(rng, n * experts, 13),
    }
}

/// The f32 cases: 24 rows of 2048 values each, one expert.
#[cfg(feature = "gpu")]
fn f32_cases(rng: &mut Rng) -> Vec<Case> {
    let (n, rows) = (2048, 24);
    let mut out = Vec::new();
    for (name, zero_every) in [
        ("uniform", 0),
        ("sum4", 0),
        ("zero-subs", 0),
        ("tiny-subs", 0),
        ("spikes", 0),
        ("zero-weights", 11),
        ("tied-zero-weights", 11),
    ] {
        let x: Vec<f32> = (0..n * rows)
            .map(|i| {
                let r = rng.next();
                match name {
                    "uniform" => (2.0 * rng.unit() - 1.0) * 0.05,
                    "sum4" => (0..4).map(|_| rng.unit() - 0.5).sum::<f32>() * 0.02,
                    "zero-subs" if (i / 32) % 5 == 0 => 0.0,
                    "zero-subs" => ((r % 7) as f32 - 3.0) * 1e-3,
                    "tiny-subs" if (i / 32) % 3 == 1 => 1e-14,
                    "tiny-subs" => ((r % 5) as f32 - 2.0) * 0.25,
                    "spikes" if i % 8 == 0 => (2.0 * rng.unit() - 1.0) * 10.0,
                    "tied-zero-weights" => {
                        let b = (r % 1000) as f32 / 1000.0;
                        let b = if r & 1 == 0 { -b } else { b };
                        b * if i % 8 == 0 { 10.0 } else { 0.1 }
                    }
                    _ => (2.0 * rng.unit() - 1.0) * 0.1,
                }
            })
            .collect();
        out.push(Case {
            name,
            ty: GgmlType::F32,
            src: x.iter().flat_map(|v| v.to_le_bytes()).collect(),
            n_per_row: n,
            rows,
            rows_per_expert: rows,
            qw: importance(rng, n, zero_every),
        });
    }
    out
}

/// ggml's bytes of `case` quantized to `kind`, through `iq_quant_ref`.
#[cfg(feature = "gpu")]
fn oracle(bin: &Path, dir: &Path, case: &Case, kind: IqKind) -> Result<Vec<u8>, GateError> {
    let dst: u32 = match kind {
        IqKind::Iq2Xxs => 16,
        IqKind::Iq1S => 19,
    };
    let src: u32 = match case.ty {
        GgmlType::F32 => 0,
        _ => 39,
    };
    let mut buf = b"IQQ1".to_vec();
    buf.extend_from_slice(&dst.to_le_bytes());
    buf.extend_from_slice(&src.to_le_bytes());
    for v in [
        case.n_per_row,
        case.rows,
        case.rows_per_expert,
        case.qw.len() / case.n_per_row,
    ] {
        buf.extend_from_slice(&(v as u64).to_le_bytes());
    }
    buf.extend_from_slice(&case.src);
    case.qw
        .iter()
        .for_each(|v| buf.extend_from_slice(&v.to_le_bytes()));
    let (inp, out) = (dir.join("in.bin"), dir.join("out.bin"));
    std::fs::write(&inp, &buf)?;
    let st = std::process::Command::new(bin)
        .arg(&inp)
        .arg(&out)
        .status()?;
    if !st.success() {
        return Err(format!("{} {}: {kind:?}: {st}", bin.display(), case.name).into());
    }
    Ok(std::fs::read(&out)?)
}

/// The rows of `case` as f32, as the oracle widens them.
#[cfg(feature = "gpu")]
fn widen(case: &Case) -> Result<Vec<f32>, GateError> {
    let mut x = vec![0.0f32; case.n_per_row * case.rows];
    dequant_row(case.ty, &case.src, &mut x)?;
    Ok(x)
}

/// Whether `got` is `want` row for row; the first differing rows are printed.
#[cfg(feature = "gpu")]
fn same_rows(what: &str, kind: IqKind, n_per_row: usize, got: &[u8], want: &[u8]) -> bool {
    let row = n_per_row / 256 * kind.block_bytes();
    if got.len() != want.len() {
        println!(
            "  {what} {kind:?}: {} bytes, ggml {}",
            got.len(),
            want.len()
        );
        return false;
    }
    let mut bad = 0;
    for (r, (g, w)) in got.chunks(row).zip(want.chunks(row)).enumerate() {
        if let Some(p) = g.iter().zip(w).position(|(a, b)| a != b) {
            if bad < 3 {
                println!(
                    "  {what} {kind:?} row {r}: super-block {} byte {}: {:#04x}, ggml {:#04x}",
                    p / kind.block_bytes(),
                    p % kind.block_bytes(),
                    g[p],
                    w[p]
                );
            }
            bad += 1;
        }
    }
    if bad > 0 {
        println!(
            "  {what} {kind:?}: {bad} of {} rows differ",
            want.len() / row
        );
    }
    bad == 0
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), GateError> {
    let bin = data_dir().join("bin/iq_quant_ref");
    if !bin.is_file() {
        return Err(format!("no oracle at {} (just build-iq-quant-ref)", bin.display()).into());
    }
    let dir: PathBuf = std::env::temp_dir().join(format!("gate-iq-quant-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let gpu = Gpu::new()?;
    let stream = gpu.stream();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut cases = vec![mxfp4_case(&mut rng)];
    cases.extend(f32_cases(&mut rng));
    let mut ok = true;
    for kind in [IqKind::Iq1S, IqKind::Iq2Xxs] {
        let q = IqQuant::new(gpu.context(), stream, kind)?;
        let tables = GridTables::new(kind);
        for case in &cases {
            let want = oracle(&bin, &dir, case, kind)?;
            let qw = DeviceBuffer::from_host(stream, &case.qw)?;
            let t0 = std::time::Instant::now();
            let got = q.quantize(
                stream,
                case.ty,
                &case.src,
                case.n_per_row,
                0,
                case.rows_per_expert,
                &qw,
            )?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let card = same_rows(case.name, kind, case.n_per_row, &got, &want);
            let x = widen(case)?;
            let row = case.n_per_row / 256 * kind.block_bytes();
            let mut host = Vec::new();
            for r in 0..2 {
                let e = r / case.rows_per_expert;
                host.extend(quantize_row(
                    kind,
                    &x[r * case.n_per_row..(r + 1) * case.n_per_row],
                    &case.qw[e * case.n_per_row..(e + 1) * case.n_per_row],
                    tables.view(),
                )?);
            }
            let host_ok = same_rows("host", kind, case.n_per_row, &host, &want[..2 * row]);
            println!(
                "{} {} {kind:?}: {} rows x {}: card = ggml {}, host rule = ggml {} ({ms:.1} ms)",
                verdict(card && host_ok),
                case.name,
                case.rows,
                case.n_per_row,
                card,
                host_ok
            );
            ok &= card && host_ok;
        }
    }
    ok &= set_clause(&bin, &dir, &mut rng)?;
    std::fs::remove_dir_all(&dir)?;
    if !ok {
        return Err(bloomery_gpu_gates::checks_failed());
    }
    println!(
        "PASSED: gate_iq_quant IQ1_S and IQ2_XXS bytes of the card equal ggml's (libggml-base of the K3.1 \
         quantization tree) on MXFP4 rows over two experts and seven f32 cases, and the host rule's on two rows of each; iq_quant_set \
         writes a set of the input's shards with the stacks ggml's and the rest copied"
    );
    Ok(())
}
