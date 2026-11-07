//! GPU gate for the card's IQ1_S and IQ2_XXS quantizer (`bloomery_gpu::iq_quant`) and the binary built on it
//! (`bloomery-quantize`): its bytes are ggml's, the quantizer llama-quantize runs, row for row.
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
//! 3. f16, bf16: four of the f32 cases rounded to F16 and to BF16 rows.
//! 4. host: `gguf::iq_quant::quantize_row` on the first two rows of every case
//!    is the same bytes, so the host rule is the card's.
//! 5. set: `bloomery-quantize` (built beside this binary) over a two-shard
//!    set — an MXFP4 and a BF16 gate stack, an MXFP4 and an F16 down stack
//!    of three experts the rules type IQ1_S and IQ2_XXS, an F32 tensor and
//!    one the rules type Q4_K — with an imatrix whose counts round to 3, to
//!    0 (importance all ones) and 5, at `--batch 2` and `--check 5`: the
//!    output's shards hold the input's metadata as llama-quantize writes it,
//!    the names, dims and order, the copied tensor's bytes, ggml's bytes for
//!    the stacks under the importance llama-quantize computes, and the host
//!    K-quant rule's bytes for the trunk weight. `--cpu` writes the same
//!    bytes as the card; `--plan` writes nothing and names every tensor; an
//!    existing output is refused.
//! 6. check: the `--check` comparison names the tensor, expert, row, super-block
//!    and byte of one flipped byte of the card's output, and passes the
//!    unflipped bytes.
//! 7. refusals: a llama plan that lacks a tensor or types a stack otherwise,
//!    an Inf or NaN in an F32, F16 or BF16 row, an MXFP4 scale byte of 255, a
//!    negative or Inf importance sum or count, and a source type the oracle
//!    does not cover each end the tool at 1 with the name in its message
//!    and no file left.
//! 8. e2e: the K3 fixture requantized to MXFP4 experts (llama-quantize), a
//!    synthetic imatrix over its stacks, then `bloomery-quantize`'s one pass
//!    (the card for the stacks, `--check 2`) and the tree's `llama-quantize`
//!    write the same file, byte for byte, with both IQ types among the
//!    stacks. The host-only gates `gate-quantize-oracle`, `gate-quantize-e2e`
//!    and `gate-quantize-plan` hold the K-quant rules, the whole file on the
//!    host and the plan.
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
use gguf::quant::{GgmlType, dequant_row, f32_to_f16_bits};
#[cfg(feature = "gpu")]
use std::path::{Path, PathBuf};

/// The clauses that run `bloomery-quantize` as a process: the set, the check, the refusals and the end to end.
#[cfg(feature = "gpu")]
mod clauses {
    use super::{
        Case, GateError, GgmlType, IqKind, Path, PathBuf, Rng, oracle, same_rows, verdict,
    };
    use gguf::Value;
    use quantize::check::compare_row;
    use std::ffi::{OsStr, OsString};

    /// A tensor of a clause's input: name, dims, type and bytes.
    type SetTensor = (String, Vec<u64>, GgmlType, Vec<u8>);

    /// Write one GGUF file of `kvs` and `tensors`.
    fn write_gguf(
        path: &Path,
        kvs: &[(String, Value)],
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

    pub(super) fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    pub(super) fn f16_bytes(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|x| super::f32_to_f16_bits(*x).to_le_bytes())
            .collect()
    }

    /// BF16 by round-to-nearest-even of the f32's top half (finite inputs).
    pub(super) fn bf16_bytes(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|x| {
                let b = x.to_bits();
                (((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16).to_le_bytes()
            })
            .collect()
    }

    /// The tool built beside this gate.
    fn tool() -> Result<PathBuf, GateError> {
        let t = std::env::current_exe()?.with_file_name("bloomery-quantize");
        if t.is_file() {
            Ok(t)
        } else {
            Err(format!("no {} (the recipe builds it beside the gate)", t.display()).into())
        }
    }

    /// One run of a process: its status code and what it printed.
    struct Ran {
        code: Option<i32>,
        out: String,
        err: String,
    }

    fn run_cmd(program: &Path, args: &[OsString], hide_cards: bool) -> Result<Ran, GateError> {
        let mut c = std::process::Command::new(program);
        c.args(args);
        if hide_cards {
            c.env("CUDA_VISIBLE_DEVICES", "");
        }
        let o = c.output()?;
        Ok(Ran {
            code: o.status.code(),
            out: String::from_utf8_lossy(&o.stdout).into_owned(),
            err: String::from_utf8_lossy(&o.stderr).into_owned(),
        })
    }

    fn os(args: &[&OsStr]) -> Vec<OsString> {
        args.iter().map(|a| (*a).to_os_string()).collect()
    }

    /// Whether no file of `dir` starts with `prefix`.
    fn no_files(dir: &Path, prefix: &str) -> Result<bool, GateError> {
        Ok(std::fs::read_dir(dir)?
            .filter_map(Result::ok)
            .all(|e| !e.file_name().to_string_lossy().starts_with(prefix)))
    }

    /// The file type every run of the tool here asks for: the stacks' and the trunk's types come from the rules.
    const FTYPE: &str = "Q4_K_M";
    const NE0: usize = 512;
    const NE1: usize = 4;
    const N_EXP: usize = 3;
    const COUNTS: [f32; 3] = [2.6, 0.3, 5.0];

    /// A routed-expert stack of the set clause: its type, the kind the rules give it and its importance.
    struct Stack {
        name: &'static str,
        ty: GgmlType,
        kind: IqKind,
        qw: Vec<f32>,
    }

    /// The set clause's input and the files its runs share.
    struct SetFix {
        first: PathBuf,
        im: PathBuf,
        rules: PathBuf,
        plan: PathBuf,
        stacks: Vec<Stack>,
    }

    fn mxfp4_bytes(rng: &mut Rng, values: usize) -> Vec<u8> {
        let mut v = Vec::new();
        for _ in 0..values / 32 {
            v.push(115 + (rng.next() % 12) as u8);
            v.extend((0..16).map(|_| rng.next() as u8));
        }
        v
    }

    fn small_values(rng: &mut Rng, n: usize) -> Vec<f32> {
        (0..n).map(|_| (rng.unit() - 0.5) * 0.1).collect()
    }

    fn stack_bytes(rng: &mut Rng, ty: GgmlType, values: usize) -> Vec<u8> {
        match ty {
            GgmlType::MXFP4 => mxfp4_bytes(rng, values),
            GgmlType::F16 => f16_bytes(&small_values(rng, values)),
            GgmlType::BF16 => bf16_bytes(&small_values(rng, values)),
            _ => f32_bytes(&small_values(rng, values)),
        }
    }

    fn plan_text(rows: &[(&str, &str, &str, u64)]) -> String {
        rows.iter()
            .map(|(n, t, d, b)| format!("{n}\t{t}\t{d}\t{b}\n"))
            .collect()
    }

    fn kvs(shard: u16, count: u16, tensors: i32) -> Vec<(String, Value)> {
        vec![
            (
                gguf::GENERAL_ARCHITECTURE.to_string(),
                Value::String("kimi-k3".to_string()),
            ),
            ("kimi-k3.block_count".to_string(), Value::U32(2)),
            ("kimi-k3.expert_count".to_string(), Value::U32(3)),
            ("split.no".to_string(), Value::U16(shard)),
            ("split.count".to_string(), Value::U16(count)),
            ("split.tensors.count".to_string(), Value::I32(tensors)),
        ]
    }

    /// The metadata an importance matrix file carries (`imatrix.datasets`, `imatrix.chunk_count`, `imatrix.chunk_size`).
    fn imatrix_kvs() -> Vec<(String, Value)> {
        vec![
            (
                "imatrix.datasets".to_string(),
                Value::Array(vec![Value::String("gate".to_string())]),
            ),
            ("imatrix.chunk_count".to_string(), Value::U32(1)),
            ("imatrix.chunk_size".to_string(), Value::U32(512)),
        ]
    }

    /// The imatrix tensors of `names` (sums over `ne0 · n_exp`, one count a expert), and the importance each gives.
    fn imatrix_tensors(
        rng: &mut Rng,
        names: &[&str],
        ne0: usize,
        counts: &[f32],
    ) -> (Vec<SetTensor>, Vec<Vec<f32>>) {
        let n_exp = counts.len();
        let (mut tensors, mut want) = (Vec::new(), Vec::new());
        for name in names {
            let sums: Vec<f32> = (0..ne0 * n_exp).map(|_| 10.0 * rng.unit()).collect();
            want.push(
                (0..ne0 * n_exp)
                    .map(|j| {
                        let c = counts[j / ne0].round();
                        if c > 0.0 { sums[j] / c } else { 1.0 }
                    })
                    .collect(),
            );
            tensors.push((
                format!("{name}.in_sum2"),
                vec![ne0 as u64, n_exp as u64],
                GgmlType::F32,
                f32_bytes(&sums),
            ));
            tensors.push((
                format!("{name}.counts"),
                vec![1, n_exp as u64],
                GgmlType::F32,
                f32_bytes(counts),
            ));
        }
        (tensors, want)
    }

    fn build_set(dir: &Path, rng: &mut Rng) -> Result<SetFix, GateError> {
        let dims = vec![NE0 as u64, NE1 as u64, N_EXP as u64];
        let stack = |rng: &mut Rng, name: &str, ty| {
            (
                name.to_string(),
                dims.clone(),
                ty,
                stack_bytes(rng, ty, NE0 * NE1 * N_EXP),
            )
        };
        let g0 = "blk.0.ffn_gate_exps.weight";
        let g1 = "blk.1.ffn_gate_exps.weight";
        let d0 = "blk.0.ffn_down_exps.weight";
        let d1 = "blk.1.ffn_down_exps.weight";
        let f32_t = |rng: &mut Rng, name: &str, dims: Vec<u64>, n| {
            (
                name.to_string(),
                dims,
                GgmlType::F32,
                f32_bytes(&small_values(rng, n)),
            )
        };
        let shards: [Vec<SetTensor>; 2] = [
            vec![
                f32_t(rng, "a.f32", vec![256, 3], 768),
                stack(rng, g0, GgmlType::MXFP4),
                stack(rng, g1, GgmlType::BF16),
            ],
            vec![
                stack(rng, d0, GgmlType::MXFP4),
                stack(rng, d1, GgmlType::F16),
                f32_t(rng, "blk.0.attn_q.weight", vec![256, 2], 512),
            ],
        ];
        let src_prefix = dir.join("set-in");
        for (i, tensors) in shards.iter().enumerate() {
            let path = PathBuf::from(format!(
                "{}-{:05}-of-00002.gguf",
                src_prefix.display(),
                i + 1
            ));
            write_gguf(&path, &kvs(i as u16, 2, 6), tensors)?;
        }
        let names = [g0, g1, d0, d1];
        let (im_tensors, want_qw) = imatrix_tensors(rng, &names, NE0, &COUNTS);
        let im = dir.join("set.imatrix.gguf");
        write_gguf(&im, &imatrix_kvs(), &im_tensors)?;
        let rules = dir.join("set.rules");
        std::fs::write(
            &rules,
            "ffn_gate_exps\\.weight=IQ1_S\nffn_down_exps\\.weight=IQ2_XXS\nattn_q\\.weight$=q4_k\n",
        )?;
        let plan = dir.join("set-plan.tsv");
        std::fs::write(
            &plan,
            plan_text(&[
                ("a.f32", "F32", "256x3", 3072),
                (g0, "IQ1_S", "512x4x3", 1200),
                (g1, "IQ1_S", "512x4x3", 1200),
                (d0, "IQ2_XXS", "512x4x3", 1584),
                (d1, "IQ2_XXS", "512x4x3", 1584),
                ("blk.0.attn_q.weight", "Q4_K", "256x2", 288),
            ]),
        )?;
        let kind = |n: &str| {
            if n.contains("gate") {
                IqKind::Iq1S
            } else {
                IqKind::Iq2Xxs
            }
        };
        let ty = |n: &str| match n {
            x if x == g1 => GgmlType::BF16,
            x if x == d1 => GgmlType::F16,
            _ => GgmlType::MXFP4,
        };
        let stacks = names
            .iter()
            .zip(want_qw)
            .map(|(n, qw)| Stack {
                name: n,
                ty: ty(n),
                kind: kind(n),
                qw,
            })
            .collect();
        Ok(SetFix {
            first: PathBuf::from(format!("{}-00001-of-00002.gguf", src_prefix.display())),
            im,
            rules,
            plan,
            stacks,
        })
    }

    fn base_args(fix: &SetFix, out: &Path) -> Vec<OsString> {
        os(&[
            fix.first.as_os_str(),
            out.as_os_str(),
            "--imatrix".as_ref(),
            fix.im.as_os_str(),
            "--tensor-type-file".as_ref(),
            fix.rules.as_os_str(),
            "--keep-split".as_ref(),
            "--allow-requantize".as_ref(),
            FTYPE.as_ref(),
        ])
    }

    fn shard_files(prefix: &Path) -> Vec<PathBuf> {
        (1..=2)
            .map(|i| PathBuf::from(format!("{}-{i:05}-of-00002.gguf", prefix.display())))
            .collect()
    }

    /// Whether the output set equals the input where it must and ggml's bytes where it quantized.
    fn verify_set(
        oracle_bin: &Path,
        dir: &Path,
        fix: &SetFix,
        out_first: &Path,
    ) -> Result<bool, GateError> {
        let input = gguf::Split::open(&fix.first)?;
        let output = gguf::Split::open(out_first)?;
        let mut ok = true;
        for i in 0..2 {
            let (a, b) = (
                input.shard(i).expect("2 shards"),
                output.shard(i).expect("2 shards"),
            );
            // llama-quantize's metadata: the input's pairs, `general.quantization_version`, `general.file_type` and
            // `quantize.imatrix.*` moved to the end (set again), the `split.*` pairs of the output shard last
            let moved = |k: &str| {
                k == "general.quantization_version"
                    || k == "general.file_type"
                    || k.starts_with("quantize.imatrix.")
                    || k.starts_with("split.")
            };
            let keep = |g: &gguf::Gguf| -> Vec<(String, Value)> {
                g.keys()
                    .iter()
                    .filter(|(k, _)| !moved(k))
                    .cloned()
                    .collect()
            };
            let tail: Vec<&str> = b
                .keys()
                .iter()
                .map(|(k, _)| k.as_str())
                .skip(keep(b).len())
                .collect();
            let want_tail = [
                "general.quantization_version",
                "general.file_type",
                "quantize.imatrix.file",
                "quantize.imatrix.dataset",
                "quantize.imatrix.entries_count",
                "quantize.imatrix.chunks_count",
                "split.no",
                "split.count",
                "split.tensors.count",
            ];
            let tail_ok = if i == 0 {
                tail == want_tail
            } else {
                tail == ["split.no", "split.count", "split.tensors.count"]
            };
            let head_ok = if i == 0 {
                keep(a) == keep(b)
            } else {
                keep(b).is_empty()
            };
            if !head_ok || !tail_ok {
                println!("  set: shard {} metadata differs: tail {tail:?}", i + 1);
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
        for s in &fix.stacks {
            let case = Case {
                name: s.name,
                ty: s.ty,
                src: input.data(s.name)?.to_vec(),
                n_per_row: NE0,
                rows: NE1 * N_EXP,
                rows_per_expert: NE1,
                qw: s.qw.clone(),
            };
            let want = oracle(oracle_bin, dir, &case, s.kind)?;
            let (_, t) = output.find(s.name).expect("names checked above");
            let want_ty = match s.kind {
                IqKind::Iq1S => GgmlType::IQ1_S,
                IqKind::Iq2Xxs => GgmlType::IQ2_XXS,
            };
            let same = same_rows("set", s.kind, NE0, output.data(s.name)?, &want);
            if t.ty != want_ty || !same {
                println!("  set: {} type {:?}, bytes = ggml {same}", s.name, t.ty);
                ok = false;
            }
        }
        if input.data("a.f32")? != output.data("a.f32")? {
            println!("  set: a.f32 is not copied byte for byte");
            ok = false;
        }
        // the trunk weight: the host K-quant rule's bytes (`gate-quantize-oracle` holds that rule to ggml)
        let src = input.data("blk.0.attn_q.weight")?;
        let vals: Vec<f32> = src
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let mut want_q = Vec::new();
        for row in vals.as_chunks::<256>().0 {
            want_q.extend(gguf::kquant::quantize_row(
                gguf::kquant::KType::Q4K,
                row,
                None,
            )?);
        }
        if output.data("blk.0.attn_q.weight")? != want_q {
            println!("  set: blk.0.attn_q.weight is not the host Q4_K rule's bytes");
            ok = false;
        }
        Ok(ok)
    }

    /// The run's refusal: status 1, `want` in its message, and no file under `prefix` left.
    fn refused(r: &Ran, want: &str, dir: &Path, prefix: &str) -> Result<bool, GateError> {
        let clean = no_files(dir, prefix)?;
        let ok = r.code == Some(1) && r.err.contains(want) && clean;
        if !ok {
            println!(
                "  refusal: want status 1, `{want}` and no file; got {:?}, files left {}, stderr: {}",
                r.code,
                !clean,
                r.err.trim()
            );
        }
        Ok(ok)
    }

    pub(super) fn tool_clauses(
        oracle_bin: &Path,
        dir: &Path,
        rng: &mut Rng,
    ) -> Result<bool, GateError> {
        let tool = tool()?;
        let fix = build_set(dir, rng)?;
        let mut all = true;
        let out_card = dir.join("set-out");
        let out_cpu = dir.join("set-cpu");
        let with = |extra: &[&str]| -> Vec<OsString> {
            let mut a = base_args(&fix, &out_card);
            a.extend(extra.iter().map(OsString::from));
            a
        };
        // the card, with the llama plan cross-checked and five rows of every stack checked against the host rule
        let plan_arg = fix.plan.to_string_lossy().into_owned();
        let r = run_cmd(
            &tool,
            &with(&["--llama-plan", &plan_arg, "--batch", "2", "--check", "5"]),
            false,
        )?;
        if r.code != Some(0) {
            println!(
                "FAIL set: bloomery-quantize ended {:?} on the whole set: {}",
                r.code,
                r.err.trim()
            );
            return Ok(false);
        }
        let first_card = shard_files(&out_card)[0].clone();
        let ok = verify_set(oracle_bin, dir, &fix, &first_card)?;
        println!(
            "{} set: bloomery-quantize over two shards (MXFP4, BF16 and F16 stacks, --check 5, the llama.cpp plan): metadata, names, dims and the F32 copy equals the input, the stacks equal ggml, the trunk weight equals the host K-quant rule",
            verdict(ok)
        );
        all &= ok;
        // --cpu: the host rule on all cores writes the card's bytes
        let r = run_cmd(
            &tool,
            &{
                let mut a = base_args(&fix, &out_cpu);
                a.extend(["--cpu", "--batch", "3"].map(OsString::from));
                a
            },
            false,
        )?;
        let same = r.code == Some(0)
            && shard_files(&out_card)
                .iter()
                .zip(shard_files(&out_cpu))
                .all(|(a, b)| matches!((std::fs::read(a), std::fs::read(b)), (Ok(x), Ok(y)) if x == y));
        if r.code != Some(0) {
            println!("  cpu: ended {:?}: {}", r.code, r.err.trim());
        }
        println!(
            "{} cpu: --cpu writes the card's bytes, file for file",
            verdict(same)
        );
        all &= same;
        // --plan writes nothing and names every tensor
        let out_plan = dir.join("set-planned");
        let r = run_cmd(
            &tool,
            &{
                let mut a = base_args(&fix, &out_plan);
                a.push("--plan".into());
                a
            },
            false,
        )?;
        let names_all = ["a.f32", "blk.0.attn_q.weight"]
            .iter()
            .chain(fix.stacks.iter().map(|s| &s.name))
            .all(|n| r.out.contains(n));
        let plan_ok = r.code == Some(0)
            && names_all
            && r.out.matches("converted on the card").count() == 4
            && r.out.matches("converted on the host").count() == 1
            && r.out.matches(": copied").count() == 1
            && r.out.contains("room: the output needs")
            && no_files(dir, "set-planned")?;
        if !plan_ok {
            println!("  plan: status {:?}, output:\n{}", r.code, r.out);
        }
        println!(
            "{} plan: --plan names every tensor with its type and fate, the derived wall and the room, and writes nothing",
            verdict(plan_ok)
        );
        all &= plan_ok;
        // an existing output is refused, not overwritten
        let r = run_cmd(&tool, &with(&[]), false)?;
        let ex_ok = r.code == Some(1) && r.err.contains("exists: not overwritten");
        println!(
            "{} exists: a run over an existing output set is refused",
            verdict(ex_ok)
        );
        all &= ex_ok;
        all &= check_clause(&first_card)?;
        all &= refusal_clauses(&tool, dir, &fix, rng)?;
        Ok(all)
    }

    /// Clause 6: `--check`'s comparison names one flipped byte of the card's output.
    fn check_clause(first_card: &Path) -> Result<bool, GateError> {
        let out = gguf::Split::open(first_card)?;
        let name = "blk.0.ffn_gate_exps.weight";
        let bytes = out.data(name)?;
        let row_bytes = NE0 / 256 * IqKind::Iq1S.block_bytes();
        let (row, at) = (5usize, 57usize);
        let want = &bytes[row * row_bytes..(row + 1) * row_bytes];
        let mut got = want.to_vec();
        let same = compare_row(name, IqKind::Iq1S, row, NE1, &got, want).is_ok();
        got[at] ^= 0x10;
        let flipped = compare_row(name, IqKind::Iq1S, row, NE1, &got, want);
        let named = matches!(
            &flipped,
            Err(quantize::Error::Mismatch { tensor, expert: 1, row: 1, block, byte, got: g, want: w })
                if tensor == name && *block == at / 50 && *byte == at % 50 && *g == want[at] ^ 0x10 && *w == want[at]
        );
        let ok = same && named;
        if !ok {
            println!("  check: equal rows pass {same}; flipped byte named {named}: {flipped:?}");
        }
        println!(
            "{} check: the --check comparison passes equal rows and names tensor, expert, row, super-block and byte of one flipped byte",
            verdict(ok)
        );
        Ok(ok)
    }

    /// One bad single-shard input: a stack, an imatrix over it, and what the refusal must say.
    struct Bad {
        tag: &'static str,
        ty: GgmlType,
        bytes: Vec<u8>,
        sums: Vec<f32>,
        counts: Vec<f32>,
        want: &'static str,
    }

    /// Clause 7: every input the tool must refuse by name.
    fn refusal_clauses(
        tool: &Path,
        dir: &Path,
        fix: &SetFix,
        rng: &mut Rng,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        // a llama plan that lacks a tensor, and one that types a stack otherwise: refused on the good set
        let short = dir.join("set-short.tsv");
        std::fs::write(
            &short,
            plan_text(&[
                ("a.f32", "F32", "256x3", 3072),
                ("blk.0.ffn_gate_exps.weight", "IQ1_S", "512x4x3", 1200),
            ]),
        )?;
        let other = dir.join("set-other.tsv");
        std::fs::write(
            &other,
            plan_text(&[
                ("a.f32", "F32", "256x3", 3072),
                ("blk.0.ffn_gate_exps.weight", "IQ2_XXS", "512x4x3", 1584),
                ("blk.1.ffn_gate_exps.weight", "IQ1_S", "512x4x3", 1200),
                ("blk.0.ffn_down_exps.weight", "IQ2_XXS", "512x4x3", 1584),
                ("blk.1.ffn_down_exps.weight", "IQ2_XXS", "512x4x3", 1584),
                ("blk.0.attn_q.weight", "Q4_K", "256x2", 288),
            ]),
        )?;
        for (plan, want) in [
            (
                &short,
                "the llama.cpp plan names 2 tensors, the set holds 6",
            ),
            (
                &other,
                "blk.0.ffn_gate_exps.weight: llama-quantize's plan says IQ2_XXS",
            ),
        ] {
            let mut a = base_args(fix, &dir.join("set-refuse"));
            a.extend(["--llama-plan".into(), plan.as_os_str().to_os_string()]);
            let r = run_cmd(tool, &a, false)?;
            ok &= refused(&r, want, dir, "set-refuse")?;
        }
        // bad inputs, one stack of two experts of two rows of 256 values
        let (ne0, ne1, n_exp) = (256usize, 2usize, 2usize);
        let name = "blk.0.ffn_gate_exps.weight";
        let n = ne0 * ne1 * n_exp;
        let sums_ok: Vec<f32> = (0..ne0 * n_exp).map(|_| 1.0 + rng.unit()).collect();
        let counts_ok = vec![3.0f32; n_exp];
        let vals = small_values(rng, n);
        let mut f32v = vals.clone();
        f32v[2 * ne0 + 5] = f32::INFINITY;
        let mut nanv = vals.clone();
        nanv[ne0 + 3] = f32::NAN;
        let mut f16b = f16_bytes(&vals);
        f16b[7 * 2..7 * 2 + 2].copy_from_slice(&0x7c00u16.to_le_bytes());
        let mut bf16b = bf16_bytes(&vals);
        bf16b[(3 * ne0 + 9) * 2..(3 * ne0 + 9) * 2 + 2].copy_from_slice(&0x7f80u16.to_le_bytes());
        let mut mx = mxfp4_bytes(rng, n);
        mx[3 * (ne0 / 32 * 17) + 3 * 17] = 255;
        let with_sum = |i: usize, v: f32| {
            let mut s = sums_ok.clone();
            s[i] = v;
            s
        };
        let bad = vec![
            Bad {
                tag: "inf32",
                ty: GgmlType::F32,
                bytes: f32_bytes(&f32v),
                sums: sums_ok.clone(),
                counts: counts_ok.clone(),
                want: "blk.0.ffn_gate_exps.weight: expert 1 row 0 value 5 is Inf (F32)",
            },
            Bad {
                tag: "nan32",
                ty: GgmlType::F32,
                bytes: f32_bytes(&nanv),
                sums: sums_ok.clone(),
                counts: counts_ok.clone(),
                want: "blk.0.ffn_gate_exps.weight: expert 0 row 1 value 3 is NaN (F32)",
            },
            Bad {
                tag: "inf16",
                ty: GgmlType::F16,
                bytes: f16b,
                sums: sums_ok.clone(),
                counts: counts_ok.clone(),
                want: "expert 0 row 0 value 7 is Inf (F16)",
            },
            Bad {
                tag: "infbf16",
                ty: GgmlType::BF16,
                bytes: bf16b,
                sums: sums_ok.clone(),
                counts: counts_ok.clone(),
                want: "expert 1 row 1 value 9 is Inf (BF16)",
            },
            Bad {
                tag: "mx255",
                ty: GgmlType::MXFP4,
                bytes: mx,
                sums: sums_ok.clone(),
                counts: counts_ok.clone(),
                want: "expert 1 row 1 value 96 is Inf (E8M0 scale byte 255)",
            },
            Bad {
                tag: "negsum",
                ty: GgmlType::F32,
                bytes: f32_bytes(&vals),
                sums: with_sum(300, -1.0),
                counts: counts_ok.clone(),
                want: "imatrix blk.0.ffn_gate_exps.weight.in_sum2: entry 300 is -1, negative",
            },
            Bad {
                tag: "infsum",
                ty: GgmlType::F32,
                bytes: f32_bytes(&vals),
                sums: with_sum(7, f32::INFINITY),
                counts: counts_ok.clone(),
                want: "imatrix blk.0.ffn_gate_exps.weight.in_sum2: entry 7 is inf, not finite",
            },
            Bad {
                tag: "negcount",
                ty: GgmlType::F32,
                bytes: f32_bytes(&vals),
                sums: sums_ok.clone(),
                counts: vec![3.0, -2.0],
                want: "imatrix blk.0.ffn_gate_exps.weight.counts: entry 1 is -2, negative",
            },
            Bad {
                tag: "q8",
                ty: GgmlType::Q8_0,
                bytes: vec![0u8; ne1 * n_exp * (ne0 / 32) * 34],
                sums: sums_ok,
                counts: counts_ok,
                want: "is Q8_0: the source types read here for IQ1_S do not include it",
            },
        ];
        let rules = dir.join("bad.rules");
        std::fs::write(&rules, "ffn_gate_exps\\.weight=IQ1_S\n")?;
        for b in bad {
            let input = dir.join(format!("bad-{}-in.gguf", b.tag));
            let im = dir.join(format!("bad-{}.imatrix.gguf", b.tag));
            let dims = vec![ne0 as u64, ne1 as u64, n_exp as u64];
            let kv = vec![
                (
                    gguf::GENERAL_ARCHITECTURE.to_string(),
                    Value::String("kimi-k3".to_string()),
                ),
                ("kimi-k3.block_count".to_string(), Value::U32(1)),
                ("kimi-k3.expert_count".to_string(), Value::U32(n_exp as u32)),
            ];
            write_gguf(&input, &kv, &[(name.to_string(), dims, b.ty, b.bytes)])?;
            let im_t: Vec<SetTensor> = vec![
                (
                    format!("{name}.in_sum2"),
                    vec![ne0 as u64, n_exp as u64],
                    GgmlType::F32,
                    f32_bytes(&b.sums),
                ),
                (
                    format!("{name}.counts"),
                    vec![1, n_exp as u64],
                    GgmlType::F32,
                    f32_bytes(&b.counts),
                ),
            ];
            write_gguf(&im, &imatrix_kvs(), &im_t)?;
            let prefix = format!("bad-{}-out", b.tag);
            let r = run_cmd(
                tool,
                &os(&[
                    input.as_os_str(),
                    dir.join(&prefix).as_os_str(),
                    "--imatrix".as_ref(),
                    im.as_os_str(),
                    "--tensor-type-file".as_ref(),
                    rules.as_os_str(),
                    "--allow-requantize".as_ref(),
                    FTYPE.as_ref(),
                ]),
                false,
            )?;
            let one = refused(&r, b.want, dir, &prefix)?;
            if !one {
                println!("  refusal `{}` failed", b.tag);
            }
            ok &= one;
        }
        println!(
            "{} refusals: a short or disagreeing llama.cpp plan, Inf and NaN rows of every source type, an MXFP4 scale byte of 255, a negative or Inf importance sum or count and an uncovered source type each end at 1 by name with no file left",
            verdict(ok)
        );
        Ok(ok)
    }

    /// The first differing key or tensor of two GGUF files, for the end-to-end clause's report.
    fn describe_diff(a: &Path, b: &Path) -> Result<Vec<String>, GateError> {
        let (ga, gb) = (gguf::Gguf::open(a)?, gguf::Gguf::open(b)?);
        let mut out = Vec::new();
        if ga.keys() != gb.keys() {
            out.push("metadata keys differ".to_string());
        }
        for t in ga.iter_tensors() {
            match gb.find(&t.name) {
                None => out.push(format!(
                    "{}: only in the llama-quantize-alone output",
                    t.name
                )),
                Some(u) if u.ty != t.ty || u.dims != t.dims => {
                    out.push(format!(
                        "{}: {:?} {:?} against {:?} {:?}",
                        t.name, t.ty, t.dims, u.ty, u.dims
                    ));
                }
                Some(u) if ga.data(t)? != gb.data(u)? => {
                    out.push(format!("{}: bytes differ", t.name))
                }
                Some(_) => {}
            }
            if out.len() >= 4 {
                break;
            }
        }
        Ok(out)
    }

    /// Clause 8: `bloomery-quantize`'s one pass equals llama-quantize's output, on the K3 fixture.
    pub(super) fn e2e_clause(dir: &Path, rng: &mut Rng) -> Result<bool, GateError> {
        let tree = PathBuf::from(std::env::var("K3Q_TREE").map_err(|_| {
            "e2e: K3Q_TREE names the K3.1 quantization tree (the recipe sets it from tools/ref/ref-paths.sh)"
        })?);
        let lq = tree.join("build/bin/llama-quantize");
        let fixture = PathBuf::from(std::env::var("K3Q_FIXTURE").unwrap_or_else(|_| {
            "/models/fixtures/kimik3/kimi-k3-fixture-00001-of-00001.gguf".to_string()
        }));
        for p in [&lq, &fixture] {
            if !p.is_file() {
                return Err(format!(
                    "e2e: no {} (just build-k3-quant-tree builds the tree; K3Q_TREE and K3Q_FIXTURE name them)",
                    p.display()
                )
                .into());
            }
        }
        let tool = tool()?;
        let work = dir.join("e2e");
        std::fs::create_dir_all(&work)?;
        let lcpp = |args: &[&OsStr]| -> Result<(), GateError> {
            let r = run_cmd(&lq, &os(args), true)?;
            if r.code == Some(0) {
                return Ok(());
            }
            let tail: Vec<&str> = r.err.lines().rev().take(6).collect();
            Err(format!(
                "e2e: llama-quantize ended {:?}: {}",
                r.code,
                tail.join(" | ")
            )
            .into())
        };
        let src = work.join("src.gguf");
        let rules_src = work.join("rules-src.txt");
        std::fs::write(&rules_src, "ffn_(down|gate|up)_exps\\.weight=mxfp4\n")?;
        let flags: [&OsStr; 4] = [
            "--output-tensor-type".as_ref(),
            "q6_k".as_ref(),
            "--token-embedding-type".as_ref(),
            "q4_k".as_ref(),
        ];
        let mut a: Vec<&OsStr> = vec![
            "--allow-requantize".as_ref(),
            "--tensor-type-file".as_ref(),
            rules_src.as_os_str(),
        ];
        a.extend(flags);
        a.extend([
            fixture.as_os_str(),
            src.as_os_str(),
            "Q4_K_M".as_ref(),
            "16".as_ref(),
        ]);
        lcpp(&a)?;
        // the stand-in's routed stacks, and an imatrix over them
        let standin = gguf::Gguf::open(&src)?;
        let stacks: Vec<(String, u64, u64)> = standin
            .iter_tensors()
            .filter(|t| t.name.ends_with("_exps.weight") && t.dims.len() >= 3)
            .map(|t| (t.name.clone(), t.dims[0], t.dims[2..].iter().product()))
            .collect();
        if stacks.len() < 2 {
            return Err(format!(
                "e2e: the stand-in holds {} routed stacks, not two or more",
                stacks.len()
            )
            .into());
        }
        let layer = stacks
            .iter()
            .filter_map(|(n, _, _)| {
                n.strip_prefix("blk.")?
                    .split('.')
                    .next()?
                    .parse::<u32>()
                    .ok()
            })
            .min()
            .ok_or("e2e: no routed stack of a block")?;
        let mut tensors: Vec<SetTensor> = Vec::new();
        for (name, ne0, n_exp) in &stacks {
            let counts: Vec<f32> = (0..*n_exp)
                .map(|e| [3.0, 0.0, 5.0, 7.0][e as usize % 4])
                .collect();
            tensors.extend(imatrix_tensors(rng, &[name], *ne0 as usize, &counts).0);
        }
        let im = work.join("e2e.imatrix.gguf");
        let im_kv = vec![
            (
                "general.type".to_string(),
                Value::String("imatrix".to_string()),
            ),
            (
                "imatrix.datasets".to_string(),
                Value::Array(vec![Value::String("gate".to_string())]),
            ),
            ("imatrix.chunk_count".to_string(), Value::U32(1)),
            ("imatrix.chunk_size".to_string(), Value::U32(512)),
        ];
        write_gguf(&im, &im_kv, &tensors)?;
        // the stand-in's weights whose rows are not whole Q4_K blocks (the fixture's narrowed 128-column tensors):
        // llama-quantize falls back to another type there and this port refuses, so both tools get one explicit Q8_0 rule
        let narrow: Vec<String> = standin
            .iter_tensors()
            .filter(|t| t.name.ends_with(".weight") && t.dims.len() >= 2 && t.dims[0] % 256 != 0)
            .map(|t| t.name.replace('.', "\\."))
            .collect();
        let narrow_rule = (!narrow.is_empty()).then(|| format!("^({})$=q8_0", narrow.join("|")));
        // llama-quantize takes the narrow rule through --tensor-type, ours through its rules file
        let base_rules = format!(
            "blk\\.{layer}\\.ffn_(gate|up)_exps\\.weight=IQ2_XXS\nffn_(down|gate|up)_exps\\.weight=IQ1_S\n"
        );
        let rules = work.join("rules.txt");
        std::fs::write(&rules, &base_rules)?;
        let rules_ours = work.join("rules-ours.txt");
        std::fs::write(
            &rules_ours,
            format!(
                "{base_rules}{}",
                narrow_rule
                    .as_ref()
                    .map_or(String::new(), |r| format!("{r}\n"))
            ),
        )?;
        let (out_a, out_b) = (work.join("alone.gguf"), work.join("ours.gguf"));
        let run_llama = |input: &Path, out: &Path| -> Result<(), GateError> {
            let mut a: Vec<&OsStr> = vec![
                "--allow-requantize".as_ref(),
                "--imatrix".as_ref(),
                im.as_os_str(),
                "--tensor-type-file".as_ref(),
                rules.as_os_str(),
            ];
            if let Some(r) = &narrow_rule {
                a.extend([OsStr::new("--tensor-type"), OsStr::new(r)]);
            }
            a.extend(flags);
            a.extend([
                input.as_os_str(),
                out.as_os_str(),
                "Q4_K_M".as_ref(),
                "16".as_ref(),
            ]);
            lcpp(&a)
        };
        run_llama(&src, &out_a)?;
        let mut ours: Vec<&OsStr> = vec![
            "--allow-requantize".as_ref(),
            "--imatrix".as_ref(),
            im.as_os_str(),
            "--tensor-type-file".as_ref(),
            rules_ours.as_os_str(),
            "--check".as_ref(),
            "2".as_ref(),
        ];
        ours.extend(flags);
        ours.extend([src.as_os_str(), out_b.as_os_str(), "Q4_K_M".as_ref()]);
        let r = run_cmd(&tool, &os(&ours), false)?;
        if r.code != Some(0) {
            println!(
                "FAIL e2e: bloomery-quantize ended {:?}: {}",
                r.code,
                r.err.trim()
            );
            return Ok(false);
        }
        let alone = gguf::Gguf::open(&out_a)?;
        let kinds = |ty: GgmlType| alone.iter_tensors().filter(|t| t.ty == ty).count();
        let (n1, n2) = (kinds(GgmlType::IQ1_S), kinds(GgmlType::IQ2_XXS));
        let same = std::fs::read(&out_a)? == std::fs::read(&out_b)?;
        if !same {
            for d in describe_diff(&out_a, &out_b)? {
                println!("  e2e: {d}");
            }
        }
        let both = n1 > 0 && n2 > 0;
        if !both {
            println!(
                "  e2e: the output holds {n1} IQ1_S and {n2} IQ2_XXS tensors: both types must be there"
            );
        }
        let ok = same && both;
        println!(
            "{} e2e: the K3 fixture's {} stacks ({n1} IQ1_S, {n2} IQ2_XXS): bloomery-quantize's one pass equals llama-quantize's output, byte for byte {same}",
            verdict(ok),
            stacks.len()
        );
        Ok(ok)
    }
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
#[derive(Clone)]
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

/// The f16 and bf16 cases: four of the f32 cases rounded to F16 and to BF16 rows.
#[cfg(feature = "gpu")]
fn half_cases(f32_cases: &[Case]) -> Vec<Case> {
    let mut out = Vec::new();
    for (from, f16, bf16) in [
        ("uniform", "f16-uniform", "bf16-uniform"),
        ("sum4", "f16-sum4", "bf16-sum4"),
        ("spikes", "f16-spikes", "bf16-spikes"),
        (
            "tied-zero-weights",
            "f16-tied-zero-weights",
            "bf16-tied-zero-weights",
        ),
    ] {
        let c = f32_cases
            .iter()
            .find(|c| c.name == from)
            .expect("an f32 case of that name");
        let x: Vec<f32> = c
            .src
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        out.push(Case {
            name: f16,
            ty: GgmlType::F16,
            src: clauses::f16_bytes(&x),
            ..c.clone()
        });
        out.push(Case {
            name: bf16,
            ty: GgmlType::BF16,
            src: clauses::bf16_bytes(&x),
            ..c.clone()
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
        GgmlType::F16 => 1,
        GgmlType::BF16 => 30,
        GgmlType::MXFP4 => 39,
        other => {
            return Err(format!("{}: the oracle has no source type {other:?}", case.name).into());
        }
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
    let f32c = f32_cases(&mut rng);
    cases.extend(half_cases(&f32c));
    cases.extend(f32c);
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
    ok &= clauses::tool_clauses(&bin, &dir, &mut rng)?;
    ok &= clauses::e2e_clause(&dir, &mut rng)?;
    std::fs::remove_dir_all(&dir)?;
    if !ok {
        return Err(bloomery_gpu_gates::checks_failed());
    }
    println!(
        "PASSED: gate_iq_quant IQ1_S and IQ2_XXS bytes of the card equal ggml's (libggml-base of the K3.1 \
         quantization tree) on MXFP4 rows over two experts, seven f32 cases and four each of f16 and bf16, and the host rule's on two rows of each; \
         bloomery-quantize writes a set of the input's shards with the stacks ggml's (--cpu the same bytes), refuses what it must by name, \
         and its output through llama-quantize equals llama-quantize alone on the K3 fixture"
    );
    Ok(())
}
