//! prepare — a Hugging Face checkpoint folder as a GGUF split the server opens as it opens any
//! (`gguf::Split`). The folder's quantization config places each tensor in a format
//! (`hf::quant_config`), the model's map names it, says how its role reads it and moves it into
//! the program's layout (`model_hf::arch::qwen35moe::Checkpoint`, qwen4exp's and Qwen3.6's reader,
//! or `model_hf::arch::qwen3moe::Checkpoint`, Qwen3-MoE's), and one table writes each format as
//! its GGUF type (`hf::modelopt::normalize`). The map is the one the config's `model_type` names,
//! among the maps `model_hf::arch::with_split_map` reads.
//!
//!     prepare <folder> <out prefix> [--shard-gib <n>] [--keep-bf16]
//!
//! writes `<out prefix>-00001-of-0000N.gguf` and on, each shard at most `n` GiB (48 by default)
//! unless one tensor with its scale and file layout is larger, the metadata in the first.
//! A quantized tensor keeps its stored format (NVFP4, MXFP8, FP8), and a BF16 matrix stays BF16,
//! the head and the token embedding among them; the server reads each through its BF16 site.
//! BF16 and F32 expert stacks are refused by name. An F32 matrix the map reads as Q8_0 is
//! refused unless `BLOOMERY_PREPARE_Q8=1` writes each single BF16 or F32 matrix read as Q8_0 as
//! ggml's reference Q8_0 instead, a lossy rewrite whose cost its lever row states; `--keep-bf16`
//! asks for the stored form and is refused beside it. Qwen3.8's decode reads about 9.27 GiB
//! of weights a token from the stored split where the Q8_0 one reads 5.73, the head alone
//! 0.55 GiB more
//! (1,271,398,400 bytes of BF16 against 34 bytes a 32 values). The output shard paths must
//! not exist. A private temporary directory holds the entire split until it passes the
//! server's reader; the first shard is published last. A persistent `<out prefix>.prepare.lock`
//! file serializes writers of that prefix. The disk must hold the split and a tenth more.
//! One line per tensor, a summary line, and exit status 1 with the error on any failure. Before
//! any shard is written, the tokenizer serving builds from the header is checked to split text
//! and stop generation where Hugging Face's does.
//!
//!     prepare <folder> --plan
//!
//! prints the plan of any folder a map reads (`model_hf::folder::open` picks the map by the
//! config's `model_type`) and writes nothing; it reads the headers and the tokenizer files only.
//! The plan is the tensors and bytes by where a load reads them and by type, the folder tensors
//! the map skips by reason, and the size of the vocabulary serving checked against Hugging Face's.
//! It is the stored form, which `--keep-bf16` names too; it takes no `--shard-gib` and refuses
//! `BLOOMERY_PREPARE_Q8=1`, whose rewrite it does not plan.

use std::error::Error;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use bloomery_levers::PREPARE_Q8;
use gguf::Value;
use gguf::write::{Layout, Writer};
use hf::modelopt::{self, Normal, Use};
use model_hf::arch::{self, MapUse};
use model_hf::folder::{self, FolderMap, Make, Planned};

const USAGE: &str = concat!(
    "usage: prepare <folder> <out prefix> [--shard-gib <n>] [--keep-bf16]\n",
    "       prepare <folder> --plan"
);

type Res<T> = Result<T, Box<dyn Error>>;

fn main() -> ExitCode {
    let ran = bloomery_levers::at_main(&[PREPARE_Q8])
        .map_err(Into::into)
        .and_then(|levers| {
            let args: Vec<String> = std::env::args().skip(1).collect();
            run(&args, levers.prepare_q8())
        });
    match ran {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("prepare: error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// The split of the folder `args` name, or with `--plan` its plan; `q8` (`BLOOMERY_PREPARE_Q8`)
/// writes its BF16 matrices the map reads as Q8_0, and its F32 matrices, in Q8_0.
fn run(args: &[String], q8: bool) -> Res<()> {
    let (mut shard_gib, mut keep_bf16, mut plan, mut paths) = (None, false, false, vec![]);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--shard-gib" => shard_gib = Some(it.next().ok_or(USAGE)?.parse::<u64>()?),
            "--keep-bf16" => keep_bf16 = true,
            "--plan" => plan = true,
            _ => paths.push(a),
        }
    }
    if plan {
        let [dir] = paths[..] else {
            return Err(USAGE.into());
        };
        if shard_gib.is_some() || q8 {
            return Err(format!(
                "--plan prints the folder's stored form and writes no shard: it takes no \
                 --shard-gib, and {PREPARE_Q8}=1 asks for a Q8_0 rewrite it does not plan"
            )
            .into());
        }
        return print_plan(Path::new(dir));
    }
    let shard_gib = shard_gib.unwrap_or(48);
    let [dir, prefix] = paths[..] else {
        return Err(USAGE.into());
    };
    if keep_bf16 && q8 {
        return Err(format!(
            "--keep-bf16 keeps the BF16 matrices {PREPARE_Q8}=1 asks to write as Q8_0: set one"
        )
        .into());
    }
    let limit = u64::checked_mul(shard_gib, 1 << 30)
        .filter(|&l| l > 0)
        .ok_or_else(|| {
            format!("--shard-gib {shard_gib}: not a shard size of 1 GiB or more in u64 bytes")
        })?;
    let dir = Path::new(dir);
    let model_type = folder::model_type(dir)?;
    arch::with_split_map(
        &model_type,
        WriteSplit {
            dir,
            prefix,
            limit,
            q8,
        },
    )
    .ok_or_else(|| {
        format!(
            "{}: model_type {model_type:?} has no split writer; prepare writes {:?}",
            dir.display(),
            arch::split_model_types()
        )
    })?
}

/// The split of a folder, written by whichever map reads it.
struct WriteSplit<'a> {
    dir: &'a Path,
    prefix: &'a str,
    /// The shard size limit in bytes.
    limit: u64,
    /// Whether a BF16 or F32 matrix is rewritten as Q8_0.
    q8: bool,
}

impl MapUse for WriteSplit<'_> {
    type Out = Res<()>;

    fn with<M: FolderMap>(self) -> Res<()> {
        write_as::<M>(self.dir, self.prefix, self.limit, self.q8)
    }
}

/// The split of the folder `dir`, read by the map `M`, written at `prefix`.
fn write_as<M: FolderMap>(dir: &Path, prefix: &str, limit: u64, q8: bool) -> Res<()> {
    let folder::Read {
        checkpoint,
        folder,
        plan,
        planned,
        kvs,
        ..
    } = folder::read::<M>(dir, q8)?;

    let tensors = planned.len();
    let shards = plan_shards(planned, &kvs, limit)?;
    let count = shards.len();
    let total = shards.iter().try_fold(0u64, |sum, (_, layout)| {
        sum.checked_add(layout.file_len())
            .ok_or("the split's file sizes overflow u64")
    })?;
    let out_dir = Path::new(prefix)
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let free = model::fileio::free_bytes(out_dir)
        .map_err(|e| format!("statvfs {}: {e}", out_dir.display()))?;
    let needed = total
        .checked_add(total / 10)
        .ok_or("the split's file sizes and free-space margin overflow u64")?;
    if free < needed {
        return Err(format!(
            "{}: {free} bytes free; the split needs {total} and a tenth more",
            out_dir.display()
        )
        .into());
    }
    println!(
        "prepare: {} tensors, {total} bytes, {count} shards; plan refused {} and left {} input scales",
        tensors,
        plan.refused.len(),
        plan.input_scales.len()
    );

    let get = |n: &str| folder.raw(n);
    let mut tensor_scales: std::collections::HashMap<String, Vec<f32>> = Default::default();
    let (shards, layouts): (Vec<_>, Vec<_>) = shards.into_iter().unzip();
    write_split(
        prefix,
        layouts,
        |no, w| {
            for p in &shards[no] {
                let t = Instant::now();
                let name = &p.decl.name;
                match &p.make {
                    Make::Scale { of } => {
                        let s = tensor_scales
                            .remove(of)
                            .ok_or_else(|| format!("{name}: {of} gave no scale"))?;
                        w.tensor(
                            name,
                            &s.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
                        )?;
                    }
                    Make::Tensor(m) if m.role == Use::Rows => {
                        // A row table's parts, one at a time, all with the table's one scale.
                        let (mut failed, mut scale) = (None, vec![]);
                        let parts = m.parts(&plan, get).map_while(|n| match n {
                            Ok(n) => {
                                scale = n.scale;
                                Some(n.bytes)
                            }
                            Err(e) => {
                                failed = Some(e);
                                None
                            }
                        });
                        let written = w.tensor_parts(name, parts);
                        if let Some(e) = failed {
                            return Err(e.into());
                        }
                        written?;
                        tensor_scales.insert(name.clone(), scale);
                    }
                    Make::Tensor(m) => {
                        let mut n: Normal = modelopt::normalize(get, &m.placed(&plan)?, m.role)?;
                        if let Some(moves) = checkpoint.mover(name, &m.steps) {
                            moves(&mut n)?;
                        }
                        if n.type_id != p.decl.type_id {
                            return Err(format!(
                                "{name}: type {}, declared {}",
                                n.type_id, p.decl.type_id
                            )
                            .into());
                        }
                        w.tensor(name, &n.bytes)?;
                        if !n.scale.is_empty() {
                            tensor_scales.insert(name.clone(), n.scale);
                        }
                    }
                }
                println!(
                    "prepare: {name} type={} bytes={} secs={:.3}",
                    p.decl.type_id,
                    p.decl.nbytes,
                    t.elapsed().as_secs_f64()
                );
            }
            Ok(())
        },
        |first| {
            // The split opens as the server opens it, its hyperparameters and layers whole.
            let split = gguf::Split::open(first)?;
            let read = model::arch::spec(&split)?;
            println!(
                "prepare: the split opens: {} layers, {} tensors",
                read.spec.layers.len(),
                split.tensor_count()
            );
            Ok(())
        },
    )
}

fn write_split(
    prefix: &str,
    layouts: Vec<Layout>,
    mut write: impl FnMut(usize, &mut Writer<BufWriter<File>>) -> Res<()>,
    validate: impl FnOnce(&Path) -> Res<()>,
) -> Res<()> {
    let lock_path = format!("{prefix}.prepare.lock");
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|e| format!("{lock_path}: open lock: {e}"))?;
    match lock.try_lock() {
        Ok(()) => {}
        Err(std::fs::TryLockError::WouldBlock) => {
            return Err(format!("{prefix}: prepare output is busy").into());
        }
        Err(std::fs::TryLockError::Error(e)) => {
            return Err(format!("{lock_path}: lock: {e}").into());
        }
    }
    let count = layouts.len();
    let paths: Vec<_> = (1..=count)
        .map(|no| PathBuf::from(format!("{prefix}-{no:05}-of-{count:05}.gguf")))
        .collect();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let stage = PathBuf::from(format!("{prefix}.part.{}.{nonce}", std::process::id()));
    let mut staged = false;
    let mut published = vec![];
    let result = (|| {
        for path in &paths {
            match std::fs::symlink_metadata(path) {
                Ok(_) => return Err(format!("{}: output already exists", path.display()).into()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(format!("{}: stat: {e}", path.display()).into()),
            }
        }
        std::fs::create_dir(&stage)?;
        staged = true;
        for (no, layout) in layouts.into_iter().enumerate() {
            let part = stage.join(paths[no].file_name().expect("a shard has a filename"));
            let mut w = Writer::new(BufWriter::new(File::create_new(&part)?), layout)?;
            write(no, &mut w)?;
            w.finish()?;
        }
        validate(&stage.join(paths[0].file_name().expect("a shard has a filename")))?;
        // The first shard is the entry point; its siblings are immutable before it appears.
        for path in paths.iter().rev() {
            let part = stage.join(path.file_name().expect("a shard has a filename"));
            model::fileio::rename_noreplace(&part, path)
                .map_err(|e| format!("{}: publish: {e}", path.display()))?;
            published.push(path);
        }
        for path in &paths {
            println!("prepare: wrote {}", path.display());
        }
        Ok(())
    })();
    let mut cleanup = vec![];
    if result.is_err() {
        for path in published {
            if let Err(e) = std::fs::remove_file(path) {
                cleanup.push(format!("{}: remove published shard: {e}", path.display()));
            }
        }
    }
    if staged && let Err(e) = std::fs::remove_dir_all(&stage) {
        cleanup.push(format!("{}: remove temporary split: {e}", stage.display()));
    }
    if let Err(e) = lock.unlock() {
        cleanup.push(format!("{lock_path}: unlock: {e}"));
    }
    if cleanup.is_empty() {
        result
    } else {
        let cleanup = cleanup.join("; ");
        match result {
            Ok(()) => Err(format!("split published; cleanup failed: {cleanup}").into()),
            Err(e) => Err(format!("{e}; cleanup failed: {cleanup}").into()),
        }
    }
}

/// The plan of the folder `dir`, whichever map reads it: headers and tokenizer, no tensor bytes.
fn print_plan(dir: &Path) -> Res<()> {
    let opened = model_hf::folder::open(dir)?;
    for line in model_hf::folder::plan_lines(&opened.header, &opened.skipped) {
        println!("prepare: plan {line}");
    }
    println!(
        "prepare: plan tokenizer: {} tokens, its specials and stop set as Hugging Face's",
        opened.tokenizer.n_vocab()
    );
    Ok(())
}

fn plan_shards<S>(
    planned: Vec<Planned<S>>,
    kvs: &[(String, Value)],
    limit: u64,
) -> Res<Vec<(Vec<Planned<S>>, Layout)>> {
    let tensors32 =
        i32::try_from(planned.len()).map_err(|_| "too many tensors for a GGUF split")?;
    let shard_keys = |no: u16, count: u16| {
        let mut keys = if no == 0 { kvs.to_vec() } else { vec![] };
        keys.extend([
            ("split.no".to_string(), Value::U16(no)),
            ("split.count".to_string(), Value::U16(count)),
            ("split.tensors.count".to_string(), Value::I32(tensors32)),
        ]);
        keys
    };
    let mut shards: Vec<Vec<Planned<S>>> = vec![vec![]];
    let mut pending = planned.into_iter().peekable();
    while let Some(p) = pending.next() {
        let mut group = vec![p];
        if pending
            .peek()
            .is_some_and(|p| matches!(p.make, Make::Scale { .. }))
        {
            group.push(pending.next().expect("the scale was just inspected"));
        }
        let no = shards.len() - 1;
        let no16 = u16::try_from(no).map_err(|_| "too many shards for a GGUF split")?;
        let last = &mut shards[no];
        let declarations = last.iter().chain(&group).map(|p| p.decl.clone()).collect();
        // split.count has the same U16 width for every final count.
        let candidate = Layout::new(&shard_keys(no16, 1), declarations)?;
        if !last.is_empty() && candidate.file_len() > limit {
            shards.push(group);
        } else {
            last.extend(group);
        }
    }
    let count16 = u16::try_from(shards.len()).map_err(|_| "too many shards for a GGUF split")?;
    shards
        .into_iter()
        .enumerate()
        .map(|(no, shard)| {
            let keys = shard_keys(u16::try_from(no)?, count16);
            let layout = Layout::new(&keys, shard.iter().map(|p| p.decl.clone()).collect())?;
            Ok((shard, layout))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use gguf::write::TensorDecl;
    use gguf::{GENERAL_ARCHITECTURE, GgmlType};
    use hf::quant_config::{Config, Format};
    use model_hf::arch::qwen35moe::Checkpoint;
    use model_hf::arch::qwen35moe::folder::{plan_tensors, served_as_hf};
    use safetensors::Dtype;
    fn split_fixture(count: usize) -> Vec<super::Layout> {
        (0..count)
            .map(|no| {
                super::Layout::new(
                    &[
                        ("split.no".into(), gguf::Value::U16(no as u16)),
                        ("split.count".into(), gguf::Value::U16(count as u16)),
                        ("split.tensors.count".into(), gguf::Value::I32(count as i32)),
                    ],
                    vec![TensorDecl {
                        name: format!("t{no}"),
                        dims: vec![2],
                        type_id: 0,
                        nbytes: 8,
                    }],
                )
                .unwrap()
            })
            .collect()
    }

    fn split_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("prepare-{tag}-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        dir
    }

    #[test]
    fn split_failure_preserves_existing_shards() {
        use super::*;
        let dir = split_dir("failure");
        let prefix = dir.join("model");
        let prefix = prefix.to_str().unwrap();
        write_split(
            prefix,
            split_fixture(2),
            |no, w| {
                w.tensor(&format!("t{no}"), &[1; 8])?;
                Ok(())
            },
            |first| {
                gguf::Split::open(first)?;
                Ok(())
            },
        )
        .unwrap();
        let paths: Vec<_> = (1..=2)
            .map(|no| format!("{prefix}-{no:05}-of-00002.gguf"))
            .collect();
        let before: Vec<_> = paths.iter().map(|p| std::fs::read(p).unwrap()).collect();
        let result = write_split(
            prefix,
            split_fixture(2),
            |no, w| {
                if no == 1 {
                    return Err("later shard failed".into());
                }
                w.tensor(&format!("t{no}"), &[2; 8])?;
                Ok(())
            },
            |_| Ok(()),
        );
        assert!(result.is_err());
        for (path, before) in paths.iter().zip(before) {
            assert_eq!(std::fs::read(path).unwrap(), before, "{path}");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn split_concurrent_writers_cannot_change_a_successful_output() {
        use super::*;
        let dir = split_dir("concurrent");
        let prefix = dir.join("model");
        let prefix = prefix.to_str().unwrap();
        let path = format!("{prefix}-00001-of-00001.gguf");
        let mut other = None;
        let first = write_split(
            prefix,
            split_fixture(1),
            |_, w| {
                let second = write_split(
                    prefix,
                    split_fixture(1),
                    |_, w| {
                        w.tensor("t0", &[2; 8])?;
                        Ok(())
                    },
                    |_| Ok(()),
                );
                other = Some(second.map(|()| std::fs::read(&path).unwrap()));
                w.tensor("t0", &[1; 8])?;
                Ok(())
            },
            |_| Ok(()),
        );
        match other.unwrap() {
            Ok(bytes) => assert_eq!(std::fs::read(&path).unwrap(), bytes),
            Err(e) => {
                assert!(e.to_string().contains("busy"), "{e}");
                first.unwrap();
            }
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn split_is_complete_and_validated_before_its_first_shard_appears() {
        use super::*;
        for count in [1, 2, 4] {
            let dir = split_dir(&format!("complete-{count}"));
            let prefix = dir.join("model");
            let prefix = prefix.to_str().unwrap();
            let first = format!("{prefix}-00001-of-{count:05}.gguf");
            write_split(
                prefix,
                split_fixture(count),
                |no, w| {
                    assert!(!Path::new(&first).exists());
                    w.tensor(&format!("t{no}"), &[no as u8; 8])?;
                    Ok(())
                },
                |staged| {
                    assert!(!Path::new(&first).exists());
                    let split = gguf::Split::open(staged)?;
                    assert_eq!(split.shard_count(), count);
                    assert_eq!(split.tensor_count(), count);
                    Ok(())
                },
            )
            .unwrap();
            let split = gguf::Split::open(&first).unwrap();
            assert_eq!(split.shard_count(), count);
            for (no, layout) in split_fixture(count).into_iter().enumerate() {
                let mut expected = Writer::new(vec![], layout).unwrap();
                expected.tensor(&format!("t{no}"), &[no as u8; 8]).unwrap();
                assert_eq!(
                    std::fs::read(split.shard_path(no).unwrap()).unwrap(),
                    expected.finish().unwrap()
                );
            }
            std::fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn split_write_validation_and_publication_errors_clean_only_their_own_files() {
        use super::*;
        for (case, failed_shard, invalid, conflict) in [
            ("write-first", Some(0), false, None),
            ("write-later", Some(1), false, None),
            ("validate", None, true, None),
            ("publish-first", None, false, Some(1)),
            ("publish-last", None, false, Some(0)),
        ] {
            let dir = split_dir(case);
            let prefix = dir.join("model");
            let prefix = prefix.to_str().unwrap();
            let paths: Vec<_> = (1..=2)
                .map(|no| format!("{prefix}-{no:05}-of-00002.gguf"))
                .collect();
            let result = write_split(
                prefix,
                split_fixture(2),
                |no, w| {
                    if failed_shard == Some(no) {
                        return Err("injected write failure".into());
                    }
                    w.tensor(&format!("t{no}"), &[1; 8])?;
                    Ok(())
                },
                |first| {
                    gguf::Split::open(first)?;
                    if invalid {
                        return Err("injected validation failure".into());
                    }
                    if let Some(no) = conflict {
                        std::fs::write(&paths[no], b"external file")?;
                    }
                    Ok(())
                },
            );
            assert!(result.is_err(), "{case}");
            for (no, path) in paths.iter().enumerate() {
                if conflict == Some(no) {
                    assert_eq!(std::fs::read(path).unwrap(), b"external file", "{case}");
                } else {
                    assert!(!Path::new(path).exists(), "{case}: {path}");
                }
            }
            for entry in std::fs::read_dir(&dir).unwrap() {
                let name = entry.unwrap().file_name();
                assert!(
                    !name.to_string_lossy().contains(".part"),
                    "{case}: {name:?}"
                );
            }
            std::fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn shard_budget_counts_scales_metadata_and_padding() {
        use super::*;

        let config: serde_json::Value = serde_json::from_reader(flate2::read::GzDecoder::new(
            &include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"),
                "/../hf/tests/fixtures/quant-configs/nvidia__Qwen3.8-Flash-Next-NVFP4__config.json.gz"))[..],
        )).unwrap();
        let checkpoint = Checkpoint::parse(&config).unwrap();
        let shape = [
            config["text_config"]["vocab_size"].as_u64().unwrap() as usize,
            config["text_config"]["hidden_size"].as_u64().unwrap() as usize,
        ];
        let kvs = vec![
            (
                GENERAL_ARCHITECTURE.into(),
                Value::String("qwen4exp".into()),
            ),
            ("general.description".into(), Value::String("x".repeat(33))),
        ];
        for (limit, lengths) in [
            (640, vec![480, 352]),
            (704, vec![704]),
            (479, vec![480, 352]),
        ] {
            let mapped = checkpoint
                .map([
                    ("lm_head.weight", shape.as_slice()),
                    ("model.language_model.embed_tokens.weight", shape.as_slice()),
                ])
                .unwrap();
            assert_eq!(mapped.len(), 2);
            let mut planned = vec![];
            for (stem, m) in ["a", "b"].into_iter().zip(mapped.into_values()) {
                let name = format!("{stem}.weight");
                planned.push(Planned {
                    decl: TensorDecl {
                        name: name.clone(),
                        dims: vec![32],
                        type_id: 0,
                        nbytes: 128,
                    },
                    make: Make::Tensor(m),
                });
                planned.push(Planned {
                    decl: TensorDecl {
                        name: format!("{stem}.scale"),
                        dims: vec![1],
                        type_id: 0,
                        nbytes: 4,
                    },
                    make: Make::Scale { of: name },
                });
            }
            let shards = plan_shards(planned, &kvs, limit).unwrap();
            let mut written = vec![];
            for (shard, layout) in shards {
                let file_len = layout.file_len();
                let mut writer = Writer::new(vec![], layout).unwrap();
                for p in &shard {
                    writer
                        .tensor(&p.decl.name, &vec![0; p.decl.nbytes as usize])
                        .unwrap();
                }
                let bytes = writer.finish().unwrap();
                assert_eq!(bytes.len() as u64, file_len);
                eprintln!("budget={limit} file_len={file_len} tensors={}", shard.len());
                assert!(
                    file_len <= limit || shard.len() == 2,
                    "{file_len} > {limit}: {} tensors",
                    shard.len()
                );
                written.push(file_len);
            }
            assert_eq!(written, lengths, "budget={limit}");
        }
    }

    /// The HF contract is exact, including when the legacy vocabulary guesses extra roles.
    #[test]
    fn the_served_tokenizer_splits_and_stops_as_hf() {
        use super::*;

        let stop_key = "tokenizer.bloomery.eog_token_ids";
        for (second, added, stop, explicit, refused) in [
            ("b", vec![2], vec![2], false, None),
            (
                "</s>",
                vec![2],
                vec![2],
                false,
                Some("splits text on tokens [1, 2]"),
            ),
            ("b", vec![2], vec![2, 0], false, Some("stops at token 0")),
            (
                "<|endoftext|>",
                vec![1, 2],
                vec![2],
                false,
                Some("serving stops at token 1"),
            ),
            (
                "<|endoftext|>",
                vec![1, 2],
                vec![1, 2],
                false,
                Some("control/rendering at token 1"),
            ),
            (
                "<|channel|>",
                vec![1, 2],
                vec![2],
                false,
                Some("control/rendering at token 1"),
            ),
            ("", vec![2], vec![2], true, Some("tokens[1] \"\"")),
            ("▁", vec![2], vec![2], true, Some("tokens[1] \"▁\"")),
            ("Ġx", vec![1, 2], vec![2], true, Some("tokens[1] \"Ġx\"")),
            ("▁x", vec![1, 2], vec![2], true, None),
            ("<|endoftext|>", vec![1, 2], vec![2], true, None),
            ("<|fim_pad|>", vec![1, 2], vec![2], true, None),
            ("<|repo_name|>", vec![1, 2], vec![2], true, None),
            ("<|file_sep|>", vec![1, 2], vec![2], true, None),
            ("<|channel|>", vec![1, 2], vec![2], true, None),
        ] {
            let second_type =
                if second == "<|channel|>" || (second == "<|endoftext|>" && stop == [2]) {
                    3
                } else if added.contains(&1) {
                    4
                } else {
                    1
                };
            let mut keys = vec![
                ("tokenizer.ggml.model".into(), Value::String("gpt2".into())),
                ("tokenizer.ggml.pre".into(), Value::String("qwen35".into())),
                (
                    "tokenizer.ggml.tokens".into(),
                    Value::Array(
                        ["a", second, "<|im_end|>", "aa"]
                            .map(|t| Value::String(t.into()))
                            .to_vec(),
                    ),
                ),
                (
                    "tokenizer.ggml.token_type".into(),
                    Value::Array([1, second_type, 3, 1].map(Value::I32).to_vec()),
                ),
                (
                    "tokenizer.ggml.merges".into(),
                    Value::Array(vec![Value::String("a a".into())]),
                ),
                ("tokenizer.ggml.eos_token_id".into(), Value::U32(2)),
                ("tokenizer.ggml.bos_token_id".into(), Value::U32(0)),
            ];
            if explicit {
                keys.push((
                    stop_key.into(),
                    Value::Array(stop.iter().copied().map(Value::U32).collect()),
                ));
            }
            let vocab = hf::vocab::Keys {
                keys: keys.clone(),
                added,
                stop,
            };
            match (served_as_hf(&keys, &vocab).map(|_| ()), refused) {
                (Ok(()), None) => {}
                (Err(e), Some(what)) if e.to_string().contains(what) => {}
                (other, _) => panic!("{second} {:?}: {other:?}", vocab.stop),
            }
            if explicit && refused.is_none() {
                for (key, value, error) in [
                    (
                        stop_key,
                        Value::Array(vec![Value::U32(0), Value::U32(2)]),
                        "serving stops at token 0",
                    ),
                    (
                        stop_key,
                        Value::Array(vec![Value::U32(0)]),
                        "stops at token 2",
                    ),
                    (
                        "tokenizer.ggml.add_bos_token",
                        Value::Bool(true),
                        "add_bos_token",
                    ),
                    (
                        "tokenizer.ggml.add_eos_token",
                        Value::Bool(true),
                        "add_eos_token",
                    ),
                    ("tokenizer.ggml.bos_token_id", Value::U32(1), "bos_token_id"),
                    ("tokenizer.ggml.eos_token_id", Value::U32(0), "eos_token_id"),
                    (
                        "tokenizer.ggml.padding_token_id",
                        Value::U32(1),
                        "padding_token_id",
                    ),
                    (
                        "tokenizer.ggml.unknown_token_id",
                        Value::U32(1),
                        "unknown_token_id",
                    ),
                    (
                        "tokenizer.ggml.seperator_token_id",
                        Value::U32(1),
                        "seperator_token_id",
                    ),
                ] {
                    let mut changed = keys.clone();
                    changed.retain(|(k, _)| k != key);
                    changed.push((key.into(), value));
                    let err = served_as_hf(&changed, &vocab).err().unwrap();
                    assert!(err.to_string().contains(error), "{second} {key}: {err}");
                }
                // The EOS id is asked whatever the flags say.
                let mut absent = hf::vocab::Keys {
                    keys: keys.clone(),
                    added: vocab.added.clone(),
                    stop: vocab.stop.clone(),
                };
                absent
                    .keys
                    .retain(|(k, _)| k != "tokenizer.ggml.eos_token_id");
                let err = served_as_hf(&absent.keys, &absent).err().unwrap();
                assert!(
                    err.to_string()
                        .contains("serving requires tokenizer.ggml.eos_token_id"),
                    "{err}"
                );
                // PIN(2026-10-28): a BOS id is asked only where the vocabulary adds a BOS. It was
                // asked always, and the Qwen3-Coder folder, which names none, was refused.
                let mut absent = hf::vocab::Keys {
                    keys: keys.clone(),
                    added: vocab.added.clone(),
                    stop: vocab.stop.clone(),
                };
                absent
                    .keys
                    .retain(|(k, _)| k != "tokenizer.ggml.bos_token_id");
                let served = served_as_hf(&absent.keys, &absent).unwrap();
                assert_eq!(served.specials().bos, None, "{second}");
                absent
                    .keys
                    .push(("tokenizer.ggml.add_bos_token".into(), Value::Bool(true)));
                let err = served_as_hf(&absent.keys, &absent).err().unwrap();
                assert!(
                    err.to_string()
                        .contains("missing metadata key tokenizer.ggml.bos_token_id"),
                    "{err}"
                );
                let mut both = keys.clone();
                let mut expected = hf::vocab::Keys {
                    keys: both.clone(),
                    added: vocab.added.clone(),
                    stop: vocab.stop.clone(),
                };
                both.push(("tokenizer.ggml.add_bos_token".into(), Value::Bool(true)));
                assert!(
                    served_as_hf(&both, &expected)
                        .err()
                        .unwrap()
                        .to_string()
                        .contains("add_bos_token")
                );
                expected.keys = both.clone();
                assert!(served_as_hf(&both, &expected).is_ok());
                // A flag of HF's that is not a bool is refused, not read as off.
                expected
                    .keys
                    .retain(|(k, _)| k != "tokenizer.ggml.add_bos_token");
                expected
                    .keys
                    .push(("tokenizer.ggml.add_bos_token".into(), Value::U32(1)));
                assert!(
                    served_as_hf(&both, &expected)
                        .err()
                        .unwrap()
                        .to_string()
                        .contains("add_bos_token is U32(1), not a bool")
                );
            }
        }
    }

    /// `--plan` takes the folder alone: another path, a shard size and the Q8_0 rewrite are refused
    /// before any folder is read, a folder with no config is refused by its name, and the write
    /// form's own refusals stand.
    #[test]
    fn plan_takes_a_folder_alone() {
        use super::*;

        let refused = |args: &[&str], q8| {
            let args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
            run(&args, q8).unwrap_err().to_string()
        };
        for args in [&["--plan"][..], &["--plan", "a", "b"], &["a"]] {
            let e = refused(args, false);
            assert!(e.contains("usage: prepare <folder> <out prefix>"), "{e}");
            assert!(e.contains("prepare <folder> --plan"), "{e}");
        }
        let e = refused(&["--plan", "--shard-gib", "1", "a"], false);
        assert!(e.contains("takes no --shard-gib"), "{e}");
        let e = refused(&["--plan", "a"], true);
        assert!(e.contains(PREPARE_Q8), "{e}");
        let e = refused(&["--plan", "/nonexistent-folder"], false);
        assert!(e.contains("/nonexistent-folder/config.json"), "{e}");
        let e = refused(&["a", "b", "--keep-bf16"], true);
        assert!(e.contains("--keep-bf16 keeps the BF16 matrices"), "{e}");
    }

    /// A folder whose `model_type` no writer reads is refused by name, with the types the writers
    /// read.
    #[test]
    fn a_model_type_no_writer_reads_is_refused() {
        use super::*;

        let dir = split_dir("no-writer");
        std::fs::write(dir.join("config.json"), r#"{"model_type": "llama"}"#).unwrap();
        let args = [
            dir.display().to_string(),
            dir.join("out").display().to_string(),
        ];
        let e = run(&args, false).unwrap_err().to_string();
        assert!(
            e.contains("model_type \"llama\" has no split writer"),
            "{e}"
        );
        assert!(e.contains("qwen3_moe") && e.contains("qwen3_5_moe"), "{e}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Serving's tokenizer, built from a Qwen3.8 folder's files as `prepare` builds it, encodes
    /// every text of the corpus to the ids Hugging Face's `tokenizers` gives
    /// (`hf/tests/fixtures/hf_encode.py`, which pins the `tokenizer.json` it ran on).
    #[test]
    #[ignore = "needs a Qwen3.8 folder's config.json and tokenizer files at $BLOOMERY_QWEN38_HF"]
    fn hw_the_served_tokenizer_encodes_the_corpus_as_hf() {
        use super::*;
        use sha2::{Digest, Sha256};

        let dir = PathBuf::from(
            std::env::var_os("BLOOMERY_QWEN38_HF").expect("$BLOOMERY_QWEN38_HF names the folder"),
        );
        let path = format!(
            "{}/../hf/tests/fixtures/qwen3.8-hf-encode.json.gz",
            env!("CARGO_MANIFEST_DIR")
        );
        let corpus: serde_json::Value =
            serde_json::from_reader(flate2::read::GzDecoder::new(File::open(&path).unwrap()))
                .unwrap();
        let file = std::fs::read(dir.join("tokenizer.json")).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&file)),
            corpus["sha256"].as_str().unwrap(),
            "{}: not the tokenizer.json of {}",
            dir.display(),
            corpus["source"]
        );
        let config = serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap());
        let checkpoint = Checkpoint::parse(&config.unwrap()).unwrap();
        let vocab = hf::vocab::keys(&dir, &checkpoint.vocab).unwrap();
        let served = served_as_hf(&vocab.keys, &vocab).unwrap();
        let cases = corpus["cases"].as_array().unwrap();
        assert!(cases.len() >= 2000, "{} cases in {path}", cases.len());
        let (mut ids, mut failures) = (0, Vec::new());
        for case in cases {
            let text = case["text"].as_str().unwrap();
            let want: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
            let ours = served.encode(text, false, true);
            ids += want.len();
            if ours != want {
                failures.push(format!("{text:?}: ours {ours:?}, HF {want:?}"));
            }
        }
        println!(
            "HF-ENCODE cases={} ids={ids} differences={} tokenizers={} source={}",
            cases.len(),
            failures.len(),
            corpus["tokenizers"],
            corpus["source"]
        );
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// A shard size of no bytes, or of more bytes than a u64 holds, is refused before the folder
    /// is read.
    #[test]
    fn a_shard_size_out_of_range_is_refused() {
        for n in ["0", "17179869184", "17179869185"] {
            let args = ["no-folder", "out", "--shard-gib", n].map(String::from);
            let e = super::run(&args, false).unwrap_err().to_string();
            assert!(e.contains("--shard-gib"), "{n}: {e}");
        }
    }

    /// Every tensor of layers 0, 1 and 3 and the unlayered text tensors of
    /// nvidia/Qwen3.8-Flash-Next-NVFP4 is declared in its stored format: each one unsloth's
    /// UD-Q4_K_XL holds in Q8_0, a BF16 matrix in the checkpoint (the head and the token
    /// embedding among them), is BF16, and Q8_0 only when `BLOOMERY_PREPARE_Q8` asks; every other
    /// tensor (the NVFP4 stacks, the PLE table's F8 rows, the gains, the scales) is the same
    /// either way. Router parts keep their stored F32 or BF16 type with either lever value.
    /// Other synthetic F32 matrices require the Q8_0 rewrite; BF16 and F32 routed stacks are
    /// refused with either lever value. Dropping either refusal fails these assertions.
    /// `--keep-bf16` asks for the stored form: taken alone, refused beside the lever.
    #[test]
    fn a_bf16_matrix_is_written_as_stored_unless_q8_is_asked() {
        use super::*;
        use model::arch::qwen35moe::names;
        use std::collections::BTreeMap;

        let gz = |name: &str| -> serde_json::Value {
            let path = format!("{}/../hf/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
            let file = File::open(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
            serde_json::from_reader(flate2::read::GzDecoder::new(file)).unwrap()
        };
        let config = gz("quant-configs/nvidia__Qwen3.8-Flash-Next-NVFP4__config.json.gz");
        let hf_quant =
            gz("quant-configs/nvidia__Qwen3.8-Flash-Next-NVFP4__hf_quant_config.json.gz");
        let quant = Config::parse(&config, Some(&hf_quant)).unwrap();
        let checkpoint = Checkpoint::parse(&config).unwrap();
        let header = gz("checkpoint-headers/qwen3.8-flash-next-nvfp4-layers-0-1-3.tensors.json.gz");
        let entries: Vec<(String, Dtype, Vec<usize>)> = header
            .as_object()
            .unwrap()
            .iter()
            .map(|(n, e)| {
                let shape = e["shape"].as_array().unwrap();
                (
                    n.clone(),
                    Dtype::of(e["dtype"].as_str().unwrap()).unwrap(),
                    shape.iter().map(|d| d.as_u64().unwrap() as usize).collect(),
                )
            })
            .collect();
        let unsloth: Vec<(String, Vec<u64>, u32)> = serde_json::from_value(gz(
            "checkpoint-headers/unsloth-qwen3.8-flash-next-ud-q4_k_xl-layers-0-1-3.tensors.json.gz",
        ))
        .unwrap();
        let q8_0: Vec<&str> = unsloth
            .iter()
            .filter(|(_, _, ty)| *ty == GgmlType::Q8_0.as_u32())
            .map(|(n, _, _)| n.as_str())
            .collect();
        for name in [names::output(), names::token_embd(), names::attn_output(3)] {
            assert!(q8_0.contains(&name.as_str()), "{name}");
        }
        let types = |q8| -> BTreeMap<String, GgmlType> {
            let (_, planned) = plan_tensors(&checkpoint, &quant, &entries, q8).unwrap();
            planned
                .into_iter()
                .map(|p| (p.decl.name, GgmlType::from_u32(p.decl.type_id)))
                .collect()
        };
        let (stored, rewritten) = (types(false), types(true));
        assert_eq!(
            stored.keys().collect::<Vec<_>>(),
            rewritten.keys().collect::<Vec<_>>()
        );
        for (name, ty) in &stored {
            let want = if q8_0.contains(&name.as_str()) {
                (GgmlType::BF16, GgmlType::Q8_0)
            } else {
                (*ty, *ty)
            };
            assert_eq!((*ty, rewritten[name]), want, "{name}");
        }
        for (router, shared) in [
            (Dtype::F32, Dtype::Bf16),
            (Dtype::Bf16, Dtype::F32),
            (Dtype::F32, Dtype::F32),
        ] {
            let mut mixed = entries.clone();
            let parts = [
                (
                    "model.language_model.layers.0.mlp.gate.weight",
                    names::ffn_gate_inp(0),
                    router,
                ),
                (
                    "model.language_model.layers.0.mlp.shared_expert_gate.weight",
                    names::ffn_gate_inp_shexp(0),
                    shared,
                ),
            ];
            for (source, _, dtype) in &parts {
                mixed.iter_mut().find(|(n, _, _)| n == source).unwrap().1 = *dtype;
            }
            for q8 in [false, true] {
                let (_, planned) = plan_tensors(&checkpoint, &quant, &mixed, q8).unwrap();
                for (_, target, dtype) in &parts {
                    let p = planned.iter().find(|p| p.decl.name == *target).unwrap();
                    let want = if *dtype == Dtype::F32 {
                        GgmlType::F32
                    } else {
                        GgmlType::BF16
                    };
                    assert_eq!(p.decl.type_id, want.as_u32(), "{target}, q8={q8}");
                    assert_eq!(
                        p.decl.nbytes,
                        p.decl.dims.iter().product::<u64>() * want.type_size().unwrap(),
                        "{target}, q8={q8}"
                    );
                }
            }
        }
        for (target, source) in [
            (
                names::attn_output(3),
                "model.language_model.layers.3.self_attn.o_proj.weight",
            ),
            (names::output(), "lm_head.weight"),
            (
                names::token_embd(),
                "model.language_model.embed_tokens.weight",
            ),
        ] {
            let mut f32 = entries.clone();
            f32.iter_mut().find(|(n, _, _)| n == source).unwrap().1 = Dtype::F32;
            for q8 in [false, true] {
                let result = plan_tensors(&checkpoint, &quant, &f32, q8);
                if !q8 {
                    let Err(e) = result else {
                        panic!("F32 matrix {target} must be refused");
                    };
                    let e = e.to_string();
                    assert!(
                        e.contains(source) && e.contains(&target) && e.contains(PREPARE_Q8),
                        "{e}"
                    );
                } else {
                    let (_, planned) = result.unwrap();
                    let p = planned.iter().find(|p| p.decl.name == target).unwrap();
                    assert_eq!(p.decl.type_id, 8, "{target}");
                }
            }
        }
        let bf16: Vec<_> = entries
            .iter()
            .filter_map(|(n, d, s)| {
                if n.contains("layers.0.mlp.experts.") && n.contains(".gate_proj.") {
                    n.ends_with(".weight")
                        .then(|| (n.clone(), Dtype::Bf16, Format::Nvfp4.logical(s)))
                } else {
                    Some((n.clone(), *d, s.clone()))
                }
            })
            .collect();
        let mut single = config.clone();
        single["text_config"]["num_experts"] = 1.into();
        single["text_config"]["num_experts_per_tok"] = 1.into();
        let single = Checkpoint::parse(&single).unwrap();
        for dtype in [Dtype::Bf16, Dtype::F32] {
            let mut stack = bf16.clone();
            for (n, d, _) in &mut stack {
                if n.contains("layers.0.mlp.experts.") && n.contains(".gate_proj.") {
                    *d = dtype;
                }
            }
            let source = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight";
            let one: Vec<_> = stack
                .iter()
                .filter(|(n, _, _)| {
                    n == source
                        || n == "lm_head.weight"
                        || n == "model.language_model.embed_tokens.weight"
                })
                .cloned()
                .collect();
            for (checkpoint, stack) in [(&checkpoint, &stack), (&single, &one)] {
                for q8 in [false, true] {
                    let Err(e) = plan_tensors(checkpoint, &quant, stack, q8) else {
                        panic!(
                            "{dtype:?} expert stacks must be refused for q8={q8}, entries={}",
                            stack.len()
                        );
                    };
                    let e = e.to_string();
                    assert!(
                        e.contains(source)
                            && e.contains(&names::ffn_gate_exps(0))
                            && e.contains("expert stacks are unsupported"),
                        "{e}"
                    );
                }
            }
        }
        for q8 in [false, true] {
            let args = ["no-folder", "out", "--keep-bf16"].map(String::from);
            let e = run(&args, q8).unwrap_err().to_string();
            assert_eq!(e.contains("--keep-bf16"), q8, "q8={q8}: {e}");
        }
    }
}
