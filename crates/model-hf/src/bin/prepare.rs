//! prepare — a Hugging Face checkpoint folder as a GGUF split the server opens as it opens any
//! (`gguf::Split`). The folder's quantization config places each tensor in a format
//! (`hf::quant_config`), the model's map names it, says how its role reads it and moves it into
//! the program's layout (`model_hf::arch::qwen35moe::Checkpoint`, qwen4exp's reader), and one table writes each
//! format as its GGUF type (`hf::modelopt::normalize`).
//!
//!     prepare <folder> <out prefix> [--shard-gib <n>]
//!
//! writes `<out prefix>-00001-of-0000N.gguf` and on, each shard at most `n` GiB (48 by default)
//! unless one tensor with its scale and file layout is larger, the metadata in the first.
//! Each shard is written as
//! `<name>.part` and renamed when whole. The disk must hold the split and a tenth more. The
//! written split is opened and its hyperparameters read as the server reads them. One line per
//! tensor, a summary line, and exit status 1 with the error on any failure. Before any shard is
//! written, the tokenizer serving builds from the header is checked to split text and stop
//! generation where Hugging Face's does.

use std::error::Error;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use gguf::write::{Layout, TensorDecl, Writer};
use gguf::{GgmlType, Value};
use hf::modelopt::{self, Normal, Use};
use hf::quant_config::{self, Config, Format};
use hf::safetensors::{Dtype, Folder};
use model_hf::arch::qwen35moe::{Checkpoint, Mapped};

const USAGE: &str = "usage: prepare <folder> <out prefix> [--shard-gib <n>]";

type Res<T> = Result<T, Box<dyn Error>>;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("prepare: error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// One tensor of the split: its declaration and how its bytes are made.
struct Planned {
    decl: TensorDecl,
    make: Make,
}

enum Make {
    /// The mapped tensor, from its sources as the plan places them.
    Tensor(Mapped),
    /// The `.scale` beside the tensor `of`: its tensor scales, one a source for a stack.
    Scale { of: String },
}

fn run(args: &[String]) -> Res<()> {
    let (dir, prefix, shard_gib) = match args {
        [dir, prefix] => (dir, prefix, 48),
        [dir, prefix, flag, n] if flag == "--shard-gib" => (dir, prefix, n.parse()?),
        _ => return Err(USAGE.into()),
    };
    let limit = u64::checked_mul(shard_gib, 1 << 30)
        .filter(|&l| l > 0)
        .ok_or_else(|| {
            format!("--shard-gib {shard_gib}: not a shard size of 1 GiB or more in u64 bytes")
        })?;
    let dir = Path::new(dir);
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let checkpoint = Checkpoint::parse(&config)?;
    let folder = Folder::open(dir)?;
    let entries: Vec<(String, Dtype, Vec<usize>)> = folder
        .names()
        .map(|n| {
            folder
                .raw(n)
                .map(|r| (n.to_string(), r.dtype, r.shape.to_vec()))
        })
        .collect::<Result<_, _>>()?;
    let plan = quant_config::plan(
        &Config::read(dir)?,
        entries
            .iter()
            .map(|(n, d, s)| (n.as_str(), *d, s.as_slice())),
    );
    let logical: Vec<(&str, Vec<usize>)> = entries
        .iter()
        .filter_map(
            |(n, _, s)| match (plan.placed.get(n), plan.refused.contains_key(n)) {
                (Some(p), _) => Some((n.as_str(), p.format.logical(s))),
                (None, true) => Some((n.as_str(), s.clone())),
                (None, false) => None,
            },
        )
        .collect();
    let mapped = checkpoint.map(logical.iter().map(|(n, s)| (*n, s.as_slice())))?;

    let mut planned = Vec::new();
    for (name, m) in mapped {
        let source = &m.sources[0];
        let format = modelopt::format_of(&m.placed(&plan)?)?;
        let type_id = modelopt::type_of(format, m.role).ok_or_else(|| {
            format!(
                "{source}: {format} read as {:?} has no GGUF type here",
                m.role
            )
        })?;
        let nbytes = size_of(type_id, &m.dims).ok_or_else(|| {
            format!(
                "{name}: dims {:?} are not whole blocks of type {type_id}",
                m.dims
            )
        })?;
        let scales = match (format, m.role) {
            (Format::Nvfp4, _) => Some(if m.sources.len() > 1 {
                m.sources.len()
            } else {
                1
            }),
            (_, Use::Rows) => Some(1),
            _ => None,
        };
        let decl = TensorDecl {
            name: name.clone(),
            dims: m.dims.clone(),
            type_id,
            nbytes,
        };
        planned.push(Planned {
            decl,
            make: Make::Tensor(m),
        });
        if let Some(n) = scales {
            let n = u64::try_from(n).map_err(|_| format!("{name}: scale count exceeds u64"))?;
            let nbytes = size_of(0, &[n])
                .ok_or_else(|| format!("{name}: scale byte count overflows u64"))?;
            let stem = name.strip_suffix(".weight").unwrap_or(&name);
            let decl = TensorDecl {
                name: format!("{stem}.scale"),
                dims: vec![n],
                type_id: 0,
                nbytes,
            };
            planned.push(Planned {
                decl,
                make: Make::Scale { of: name },
            });
        }
    }

    let mut kvs = checkpoint.header(
        planned
            .iter()
            .map(|p| (p.decl.name.as_str(), p.decl.dims.as_slice())),
        &folder,
    )?;
    let vocab = hf::vocab::keys(dir, &checkpoint.vocab)?;
    kvs.extend(vocab.keys.iter().cloned());
    served_as_hf(
        &kvs,
        &vocab,
        Path::new(&format!("{prefix}.tokenizer.gguf.part")),
    )?;

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
    for (no, (shard, layout)) in shards.into_iter().enumerate() {
        let path = PathBuf::from(format!("{prefix}-{:05}-of-{count:05}.gguf", no + 1));
        let part = path.with_extension("gguf.part");
        let mut w = Writer::new(BufWriter::new(File::create(&part)?), layout)?;
        for p in shard {
            let t = Instant::now();
            let name = &p.decl.name;
            match p.make {
                Make::Scale { of } => {
                    let s = tensor_scales
                        .remove(&of)
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
                    for step in &m.steps {
                        step.apply(&checkpoint.geometry, name, &mut n)?;
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
        w.finish()?;
        std::fs::rename(&part, &path)?;
        println!("prepare: wrote {}", path.display());
    }
    // The split opens as the server opens it, its hyperparameters and layers whole.
    let split = gguf::Split::open(format!("{prefix}-00001-of-{count:05}.gguf"))?;
    let read = model::arch::spec(&split)?;
    println!(
        "prepare: the split opens: {} layers, {} tensors",
        read.spec.layers.len(),
        split.tensor_count()
    );
    Ok(())
}

fn plan_shards(
    planned: Vec<Planned>,
    kvs: &[(String, Value)],
    limit: u64,
) -> Res<Vec<(Vec<Planned>, Layout)>> {
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
    let mut shards: Vec<Vec<Planned>> = vec![vec![]];
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

/// Serving's tokenizer, which must preserve HF's split, rendering, special ids, automatic BOS/EOS
/// and generation stop sets. The header-only `probe` is removed before any shard is written.
fn served_as_hf(
    kvs: &[(String, Value)],
    vocab: &hf::vocab::Keys,
    probe: &Path,
) -> Res<tokenizer::Tokenizer> {
    let open = || -> Res<tokenizer::Tokenizer> {
        let layout = Layout::new(kvs, Vec::new())?;
        Writer::new(File::create(probe)?, layout)?.finish()?;
        Ok(tokenizer::Tokenizer::from_gguf(probe)?)
    };
    let served = open();
    if probe.exists() {
        std::fs::remove_file(probe)?;
    }
    let served = served?;
    let mut split = served.special_tokens().to_vec();
    split.sort_unstable();
    if split != vocab.added {
        return Err(format!(
            "serving splits text on tokens {split:?}, HF on its added tokens {:?}",
            vocab.added
        )
        .into());
    }
    if let Some(id) = vocab.stop.iter().find(|id| !served.eog().contains(id)) {
        return Err(format!(
            "HF's generate stops at token {id}, where serving does not end generation"
        )
        .into());
    }
    if let Some(id) = served.eog().iter().find(|id| !vocab.stop.contains(id)) {
        return Err(format!("serving stops at token {id}, where HF's generate continues").into());
    }
    let expected = |key: &str| vocab.keys.iter().find(|(k, _)| k == key).map(|(_, v)| v);
    let Some(Value::Array(types)) = expected("tokenizer.ggml.token_type") else {
        return Err("HF tokenizer has no tokenizer.ggml.token_type".into());
    };
    if types.len() != served.n_vocab() {
        return Err("serving vocabulary size differs from HF's".into());
    }
    for (id, ty) in types.iter().enumerate() {
        let id = u32::try_from(id)?;
        let control = served
            .attr(id)
            .is_some_and(|a| a & (tokenizer::attr::CONTROL | tokenizer::attr::UNKNOWN) != 0);
        if control != (*ty == Value::I32(3)) {
            return Err(
                format!("serving control/rendering at token {id} differs from HF's").into(),
            );
        }
    }
    for (key, actual) in [
        ("tokenizer.ggml.add_bos_token", served.add_bos()),
        ("tokenizer.ggml.add_eos_token", served.add_eos()),
    ] {
        if actual != expected(key).and_then(Value::as_bool).unwrap_or(false) {
            return Err(format!("serving {key} differs from HF's").into());
        }
    }
    let specials = served.specials();
    for (key, actual) in [
        ("tokenizer.ggml.bos_token_id", specials.bos),
        ("tokenizer.ggml.eos_token_id", specials.eos),
        ("tokenizer.ggml.padding_token_id", specials.pad),
        ("tokenizer.ggml.unknown_token_id", specials.unk),
        ("tokenizer.ggml.seperator_token_id", specials.sep),
    ] {
        if actual.is_none()
            && matches!(
                key,
                "tokenizer.ggml.bos_token_id" | "tokenizer.ggml.eos_token_id"
            )
        {
            return Err(format!("serving requires {key}").into());
        }
        if actual.map(u64::from) != expected(key).and_then(Value::as_u64) {
            return Err(format!("serving {key} differs from HF's").into());
        }
    }
    Ok(served)
}

/// The bytes of a tensor of `type_id` and `dims`, its blocks as the type's row sizes them.
fn size_of(type_id: u32, dims: &[u64]) -> Option<u64> {
    let ty = GgmlType::from_u32(type_id);
    let (blck, size) = ty.blck_size().zip(ty.type_size())?;
    if dims.is_empty() || !dims[0].is_multiple_of(blck) {
        return None;
    }
    gguf::tensor_bytes(blck, size, dims)
}

#[cfg(test)]
mod tests {
    use gguf::GENERAL_ARCHITECTURE;
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

        let probe =
            std::env::temp_dir().join(format!("prepare-served-{}.gguf", std::process::id()));
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
            match (served_as_hf(&keys, &vocab, &probe).map(|_| ()), refused) {
                (Ok(()), None) => {}
                (Err(e), Some(what)) if e.to_string().contains(what) => {}
                (other, _) => panic!("{second} {:?}: {other:?}", vocab.stop),
            }
            assert!(!probe.exists());
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
                    let err = served_as_hf(&changed, &vocab, &probe).err().unwrap();
                    assert!(err.to_string().contains(error), "{second} {key}: {err}");
                    assert!(!probe.exists());
                }
                for key in ["tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"] {
                    let mut absent = hf::vocab::Keys {
                        keys: keys.clone(),
                        added: vocab.added.clone(),
                        stop: vocab.stop.clone(),
                    };
                    absent.keys.retain(|(k, _)| k != key);
                    let err = served_as_hf(&absent.keys, &absent, &probe).err().unwrap();
                    assert!(
                        err.to_string().contains(&format!("serving requires {key}")),
                        "{err}"
                    );
                    assert!(!probe.exists());
                }
                let mut both = keys.clone();
                let mut expected = hf::vocab::Keys {
                    keys: both.clone(),
                    added: vocab.added.clone(),
                    stop: vocab.stop.clone(),
                };
                both.push(("tokenizer.ggml.add_bos_token".into(), Value::Bool(true)));
                assert!(
                    served_as_hf(&both, &expected, &probe)
                        .err()
                        .unwrap()
                        .to_string()
                        .contains("add_bos_token")
                );
                expected.keys = both.clone();
                assert!(served_as_hf(&both, &expected, &probe).is_ok());
            }
        }
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
        let probe =
            std::env::temp_dir().join(format!("prepare-corpus-{}.gguf", std::process::id()));
        let served = served_as_hf(&vocab.keys, &vocab, &probe).unwrap();
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
            let e = super::run(&args).unwrap_err().to_string();
            assert!(e.contains("--shard-gib"), "{n}: {e}");
        }
    }
}
