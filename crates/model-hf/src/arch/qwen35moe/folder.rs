//! A Qwen3.8 or Qwen3.6 ([`super::Family`]) checkpoint folder as the program reads it: the tensors
//! of its split, each declared in the GGUF type its stored format and role give
//! ([`plan_tensors`]), the header those tensors and the folder's keys make ([`read`]), and
//! serving's tokenizer checked against Hugging Face's ([`served_as_hf`]). `prepare` writes that
//! header and the tensors' bytes as a split; a load opens the folder as it is ([`open`]): the same
//! header held in memory (`gguf::Split::header_only`), the tensors a card holds read through the
//! folder's part source ([`crate::part::Checkpoint`]), and the PLE table's rows read where the
//! folder stores them (`model::part::StoredRows`). Both read the folder through these functions
//! alone, so a folder and the split `prepare` writes of it describe one model.

use std::collections::BTreeMap;
use std::path::Path;

use bloomery_levers::PREPARE_Q8;
use gguf::write::{Layout, TensorDecl};
use gguf::{GgmlType, Split, Value};
use hf::modelopt::{self, Plain, Use};
use hf::quant_config::{self, Config, Format, Plan};
use safetensors::{Dtype, Folder};

use model::part::{StoredPlane, StoredRows};

use super::{Checkpoint, Mapped};
use crate::part::{self, Made};

/// Why a folder was refused.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Map(#[from] super::Error),
    #[error(transparent)]
    Part(#[from] part::Error),
    #[error(transparent)]
    Tensors(#[from] safetensors::Error),
    #[error(transparent)]
    Quant(#[from] quant_config::Error),
    #[error(transparent)]
    Normal(#[from] modelopt::Error),
    #[error(transparent)]
    Vocab(#[from] hf::vocab::Error),
    #[error(transparent)]
    Layout(#[from] gguf::write::WriteError),
    #[error(transparent)]
    Header(#[from] gguf::LoadError),
    #[error(transparent)]
    Tokenizer(#[from] tokenizer::Error),
    #[error("{0}")]
    Refused(String),
}

impl From<String> for Error {
    fn from(why: String) -> Error {
        Error::Refused(why)
    }
}

/// One tensor of the split: its declaration and how its bytes are made.
pub struct Planned {
    pub decl: TensorDecl,
    pub make: Make,
}

/// How a planned tensor's bytes are made.
pub enum Make {
    /// The mapped tensor, from its sources as the plan places them.
    Tensor(Mapped),
    /// The `.scale` beside the tensor `of`: its tensor scales, one a source for a stack.
    Scale { of: String },
}

/// The folder as both readers take it: its config, its tensors, the quantization plan, the
/// planned tensors, the header's keys and serving's tokenizer.
pub struct Read {
    pub checkpoint: Checkpoint,
    pub folder: Folder,
    pub plan: Plan,
    pub planned: Vec<Planned>,
    pub kvs: Vec<(String, Value)>,
    pub tokenizer: tokenizer::Tokenizer,
}

/// The folder `dir` read and checked: `config.json`, its tensors placed by its quantization
/// config and planned ([`plan_tensors`], `q8` as there), the header's keys (the model's and the
/// tokenizer's), and serving's tokenizer built from them and checked against Hugging Face's
/// ([`served_as_hf`]).
pub fn read(dir: &Path, q8: bool) -> Result<Read, Error> {
    let config_path = dir.join("config.json");
    let config: serde_json::Value = std::fs::read(&config_path)
        .map_err(|e| e.to_string())
        .and_then(|b| serde_json::from_slice(&b).map_err(|e| e.to_string()))
        .map_err(|e| format!("{}: {e}", config_path.display()))?;
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
    let (plan, planned) = plan_tensors(&checkpoint, &Config::read(dir)?, &entries, q8)?;
    let mut kvs = checkpoint.header(
        planned
            .iter()
            .map(|p| (p.decl.name.as_str(), p.decl.dims.as_slice())),
        &folder,
    )?;
    let vocab = hf::vocab::keys(dir, &checkpoint.vocab)?;
    kvs.extend(vocab.keys.iter().cloned());
    let tokenizer = served_as_hf(&kvs, &vocab)?;
    Ok(Read {
        checkpoint,
        folder,
        plan,
        planned,
        kvs,
        tokenizer,
    })
}

/// A folder as a load opens it ([`open`]).
pub struct Opened {
    /// The header `prepare` writes as its split's first shard, with every tensor of the split
    /// and none of their bytes, named by the folder's path.
    pub header: Split,
    /// The MTP layer's header, borrowing the target's embedding and head ([`mtp`]): `None` for
    /// a folder that holds no MTP layer, and the layer's refusal for one the draft does not
    /// read, which refuses the draft alone.
    pub draft: Result<Option<Split>, Error>,
    /// Every tensor of the header but the PLE table and the `.scale` tensors, whose scales each
    /// part carries, and the MTP layer's tensors when the draft reads them.
    pub source: part::Checkpoint,
    /// The PLE table's rows where the folder stores them, when the model has the table.
    pub ple: Option<StoredRows>,
    /// Serving's tokenizer, checked against Hugging Face's.
    pub tokenizer: tokenizer::Tokenizer,
}

/// The folder `dir` opened for a load, every part in its stored format (no Q8_0 rewrite): its
/// header ([`Split::header_only`] of [`read`]'s keys and planned tensors), its card tensors as a
/// part source, its PLE table's rows ([`ple_rows`]) and its MTP layer ([`mtp`]). Any refusal
/// of [`read`] or [`ple_rows`] is the folder's; one of [`mtp`] is the draft's, held in
/// [`Opened::draft`].
pub fn open(dir: &Path) -> Result<Opened, Error> {
    let Read {
        checkpoint,
        folder,
        plan,
        mut planned,
        kvs,
        tokenizer,
    } = read(dir, false)?;
    let layout = Layout::new(&kvs, planned.iter().map(|p| p.decl.clone()).collect())?;
    let header = Split::header_only(dir.to_path_buf(), &layout)?;
    let draft = mtp(dir, &checkpoint, (&folder, &plan), &kvs).map(|drafted| {
        drafted.map(|(draft, tensors)| {
            planned.extend(tensors);
            draft
        })
    });
    let mut made = BTreeMap::new();
    let mut ple = None;
    for Planned { decl, make } in planned {
        let Make::Tensor(m) = make else {
            continue;
        };
        if m.role == Use::Rows {
            ple = Some(ple_rows(&folder, &plan, &decl, &m)?);
            continue;
        }
        let step = (!m.steps.is_empty()).then(|| {
            let (g, name, steps) = (checkpoint.geometry.clone(), decl.name.clone(), m.steps);
            Box::new(move |n: &mut modelopt::Normal| {
                steps
                    .iter()
                    .try_for_each(|s| s.apply(&g, &name, n))
                    .map_err(|e| e.to_string())
            }) as part::Step
        });
        made.insert(
            decl.name,
            Made {
                sources: m.sources,
                role: m.role,
                dims: m.dims,
                step,
            },
        );
    }
    Ok(Opened {
        header,
        draft,
        source: part::Checkpoint::new(folder, plan, made)?,
        ple,
        tokenizer,
    })
}

/// The MTP layer of the folder `dir` as the draft reads it, every tensor in its stored format:
/// its header ([`draft_header`] over the target's keys `kvs`) and its planned tensors, `None`
/// for a folder that holds no `mtp.` tensor. A layer the map or the planner refuses (a second
/// MTP layer, an F32 projection, an unquantized expert stack) is refused by name.
fn mtp(
    dir: &Path,
    checkpoint: &Checkpoint,
    (folder, plan): (&Folder, &Plan),
    kvs: &[(String, Value)],
) -> Result<Option<(Split, Vec<Planned>)>, Error> {
    let logical = folder
        .names()
        .filter(|n| n.starts_with("mtp."))
        .filter_map(|n| {
            if let Some(p) = plan.placed.get(n) {
                Some(
                    folder
                        .raw(n)
                        .map(|r| (n.to_string(), p.format.logical(r.shape))),
                )
            } else if plan.refused.contains_key(n) {
                Some(folder.raw(n).map(|r| (n.to_string(), r.shape.to_vec())))
            } else {
                None
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let tensors = planned_tensors(
        plan,
        super::map(
            &checkpoint.geometry,
            logical.iter().map(|(n, s)| (n.as_str(), s.as_slice())),
            true,
        )?,
        false,
    )?;
    let header = draft_header(
        dir,
        kvs,
        checkpoint.geometry.layers,
        tensors.iter().map(|p| p.decl.clone()).collect(),
    )?;
    Ok(header.map(|h| (h, tensors)))
}

fn draft_header(
    dir: &Path,
    kvs: &[(String, Value)],
    layers: usize,
    tensors: Vec<TensorDecl>,
) -> Result<Option<Split>, Error> {
    if tensors.is_empty() {
        return Ok(None);
    }
    let mut keys = kvs.to_vec();
    for (key, value) in &mut keys {
        if key == "qwen4exp.block_count" {
            *value = Value::U32(
                u32::try_from(layers)
                    .ok()
                    .and_then(|n| n.checked_add(1))
                    .ok_or_else(|| "MTP block count exceeds u32".to_string())?,
            );
        }
        if key == "qwen4exp.attention.compress_ratios" {
            let Value::Array(ratios) = value else {
                return Err("MTP compression ratios are not an array".to_string().into());
            };
            // The shared MTP walk uses full attention and leaves the indexer unread.
            ratios.push(Value::I32(0));
        }
    }
    keys.push(("qwen4exp.nextn_predict_layers".to_string(), Value::U32(1)));
    let layout = Layout::new(&keys, tensors)?;
    Ok(Some(Split::header_only(dir.to_path_buf(), &layout)?))
}

/// The row table `decl`, mapped as `m`, where `folder` stores it: each part's byte range in its
/// shard. F8 values or NVFP4 codes and block scales stay in their separate
/// stored planes. Each part must have the table's width and tensor scale,
/// and their rows must add up to the table's.
fn ple_rows(
    folder: &Folder,
    plan: &Plan,
    decl: &TensorDecl,
    m: &Mapped,
) -> Result<StoredRows, Error> {
    let name = &decl.name;
    let ty = GgmlType::from_u32(decl.type_id);
    if !matches!(ty, GgmlType::F8E4M3_ROWS | GgmlType::NVFP4_ROWS) {
        return Err(format!("{name}: {ty} is not F8 or NVFP4 rows").into());
    }
    let (width, rows) = match decl.dims[..] {
        [width, rows]
            if width > 0 && rows > 0 && size_of(decl.type_id, &decl.dims) == Some(decl.nbytes) =>
        {
            (width, rows)
        }
        _ => return Err(format!("{name}: dims {:?} are not a table's", decl.dims).into()),
    };
    let get = |n: &str| folder.raw(n);
    let mut scale: Option<f32> = None;
    let mut runs = Vec::with_capacity(m.sources.len());
    let mut scale_runs = Vec::new();
    let divisor = if ty == GgmlType::NVFP4_ROWS { 2 } else { 1 };
    let mut held = 0u64;
    for (source, placed) in m.placed(plan)? {
        let s = if ty == GgmlType::NVFP4_ROWS {
            let (s, block_scale) = modelopt::nvfp4_rows_scale(&get, source, placed)?;
            scale_runs.push(folder.file_range(block_scale)?);
            s
        } else {
            modelopt::fp8_rows_scale(&get, source, placed)?.0
        };
        if let Some(t) = scale.filter(|&t| t != s) {
            return Err(format!("{source}: tensor scale {s}, the table's {t}").into());
        }
        scale = Some(s);
        let shape = folder.raw(source)?.shape;
        let &[part_rows, columns] = shape else {
            return Err(format!("{source}: shape {shape:?} is not a table part's").into());
        };
        let part_width = (columns as u64)
            .checked_mul(divisor)
            .ok_or_else(|| format!("{source}: row width overflow"))?;
        if part_width != width {
            return Err(
                format!("{source}: rows of {part_width} values, the table's {width}").into(),
            );
        }
        held = held
            .checked_add(part_rows as u64)
            .ok_or_else(|| format!("{name}: row count overflow"))?;
        runs.push(folder.file_range(source)?);
    }
    if held != rows {
        return Err(format!("{name}: its parts hold {held} rows, the table {rows}").into());
    }
    Ok(StoredRows {
        values: StoredPlane {
            row_bytes: width / divisor,
            runs,
        },
        block_scales: (ty == GgmlType::NVFP4_ROWS).then_some(StoredPlane {
            row_bytes: width / 16,
            runs: scale_runs,
        }),
        scale: scale.ok_or_else(|| format!("{name}: a table of no parts"))?,
    })
}

/// The folder's `entries` (name, dtype, shape) placed by its `config`, and each tensor the map
/// makes of them declared in the type its placed format and role give, its `.scale` beside it
/// where the type reads one. An unquantized matrix the map reads as Q8_0 stays in its stored
/// format unless `q8` asks for the rewrite; F32 matrices need that rewrite. Unquantized expert
/// stacks are unsupported.
pub fn plan_tensors(
    checkpoint: &Checkpoint,
    config: &Config,
    entries: &[(String, Dtype, Vec<usize>)],
    q8: bool,
) -> Result<(Plan, Vec<Planned>), Error> {
    let plan = quant_config::plan(
        config,
        entries
            .iter()
            .map(|(n, d, s)| (n.as_str(), *d, s.as_slice())),
        crate::arch::module_class,
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
    let planned = planned_tensors(&plan, mapped, q8)?;
    Ok((plan, planned))
}

fn planned_tensors(
    plan: &Plan,
    mapped: BTreeMap<String, Mapped>,
    q8: bool,
) -> Result<Vec<Planned>, Error> {
    let mut planned = Vec::new();
    for (name, mut m) in mapped {
        let source = &m.sources[0];
        let format = modelopt::format_of(&m.placed(plan)?)?;
        if m.role == Use::Matrix(Plain::Q8_0) && matches!(format, Format::Bf16 | Format::F32) {
            if m.dims.len() == 3 {
                return Err(format!(
                    "{source}: {name}: {format} expert stacks are unsupported by the converter"
                )
                .into());
            }
            if !q8 {
                if format == Format::F32 {
                    return Err(format!(
                        "{source}: {name} is stored {format}, a form its card site does not read; {PREPARE_Q8}=1 writes Q8_0"
                    )
                    .into());
                }
                m.role = Use::Matrix(Plain::Bf16);
            }
        }
        if format == Format::F32
            && (name.ends_with(".ffn_gate_inp.weight")
                || name.ends_with(".ffn_gate_inp_shexp.weight"))
        {
            m.role = Use::Matrix(Plain::F32);
        }
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
        // One tensor scale an expert of a stack; one for a row table, whatever its parts.
        let scales = GgmlType::from_u32(type_id)
            .reads_scale()
            .then_some(match m.role {
                Use::Rows => 1,
                Use::Matrix(_) => m.sources.len(),
            });
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
    Ok(planned)
}

/// Serving's tokenizer, built from the header's keys `kvs`, which must preserve HF's split,
/// rendering, special ids, automatic BOS/EOS and generation stop sets.
pub fn served_as_hf(
    kvs: &[(String, Value)],
    vocab: &hf::vocab::Keys,
) -> Result<tokenizer::Tokenizer, Error> {
    let served = tokenizer::Tokenizer::from_keys(kvs)?;
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
        return Err("HF tokenizer has no tokenizer.ggml.token_type"
            .to_string()
            .into());
    };
    if types.len() != served.n_vocab() {
        return Err("serving vocabulary size differs from HF's"
            .to_string()
            .into());
    }
    for (id, ty) in types.iter().enumerate() {
        let id = u32::try_from(id).map_err(|_| format!("token {id} past u32"))?;
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
pub fn size_of(type_id: u32, dims: &[u64]) -> Option<u64> {
    let ty = GgmlType::from_u32(type_id);
    let (blck, size) = ty.blck_size().zip(ty.type_size())?;
    if dims.is_empty() || !dims[0].is_multiple_of(blck) {
        return None;
    }
    gguf::tensor_bytes(blck, size, dims)
}

#[cfg(test)]
mod tests {
    use hf::quant_config::plan;
    use model::arch::qwen35moe::names;

    use super::*;

    /// A folder whose target is whole opens whatever its MTP layer holds: an MTP layer the draft
    /// reads is planned beside the target and its tensors join the part source; one it does not
    /// read (F32 input projections) is the draft's refusal alone, held by name, with the target's
    /// header and parts as before; a folder with no MTP layer has no draft. Mutant: the draft's
    /// refusal propagated out of `open` (the target refused with it), or its tensors kept.
    #[test]
    fn a_target_opens_whatever_its_mtp_layer_holds() {
        use super::super::tests::{small_config, tokenizer};
        let mut config = small_config();
        config
            .as_object_mut()
            .unwrap()
            .remove("quantization_config");
        let text = config["text_config"].as_object_mut().unwrap();
        text.remove("quantization_config");
        text.remove("ple_layer_ids");
        let dir = crate::test_support::temp_dir("qwen-folder-mtp");
        for (file, json) in [
            ("config.json", config),
            ("tokenizer.json", tokenizer()),
            ("tokenizer_config.json", serde_json::json!({})),
            ("generation_config.json", serde_json::json!({})),
        ] {
            std::fs::write(dir.join(file), json.to_string()).unwrap();
        }
        let table = vec![0u8; 8 * 2560 * 2];
        let write = |mtp: Option<&str>| {
            let projection = vec![0u8; 2 * 2 * if mtp == Some("F32") { 4 } else { 2 }];
            let mut tensors: Vec<(&str, &str, &[usize], &[u8])> = vec![
                (
                    "model.language_model.embed_tokens.weight",
                    "BF16",
                    &[8, 2560],
                    &table,
                ),
                ("lm_head.weight", "BF16", &[8, 2560], &table),
            ];
            if let Some(dtype) = mtp {
                for name in ["mtp.fc_embedding.weight", "mtp.fc_hidden.weight"] {
                    tensors.push((name, dtype, &[2, 2], &projection));
                }
            }
            std::fs::write(dir.join("a.safetensors"), safetensors::write(&tensors)).unwrap();
            let map: BTreeMap<&str, &str> =
                tensors.iter().map(|t| (t.0, "a.safetensors")).collect();
            std::fs::write(
                dir.join(Folder::INDEX),
                serde_json::json!({ "weight_map": map }).to_string(),
            )
            .unwrap();
            open(&dir)
        };
        let target = [names::token_embd(), names::output()];
        let held = |o: &Opened| {
            let mut names: Vec<&str> = o.source.made().keys().map(String::as_str).collect();
            names.sort_unstable();
            names.into_iter().map(str::to_owned).collect::<Vec<_>>()
        };
        let mut want = target.to_vec();
        want.sort_unstable();
        let eh_proj = "blk.48.nextn.eh_proj.weight".to_string();
        let o = write(Some("BF16")).expect("a folder with a BF16 MTP layer");
        let draft = o
            .draft
            .as_ref()
            .expect("BF16 projections")
            .as_ref()
            .expect("a draft");
        assert!(draft.find(&eh_proj).is_some());
        let mut with_draft = want.clone();
        with_draft.push(eh_proj.clone());
        with_draft.sort_unstable();
        assert_eq!(held(&o), with_draft);
        let o = write(Some("F32")).expect("the target opens beside an F32 MTP layer");
        let Err(why) = &o.draft else {
            panic!("F32 projections: a draft");
        };
        let why = why.to_string();
        assert!(why.contains("is stored F32"), "{why}");
        assert!(why.contains("mtp.fc_embedding.weight"), "{why}");
        assert_eq!(held(&o), want);
        for name in &target {
            assert!(o.header.find(name).is_some(), "{name}");
        }
        assert!(o.header.find(&eh_proj).is_none());
        let o = write(None).expect("a folder with no MTP layer");
        assert!(matches!(o.draft, Ok(None)));
        assert_eq!(held(&o), want);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn mtp_header_keeps_the_signed_ratio_array_and_the_target_header() {
        let keys = vec![
            ("qwen4exp.block_count".to_string(), Value::U32(48)),
            (
                "qwen4exp.attention.compress_ratios".to_string(),
                Value::Array(vec![Value::I32(128); 48]),
            ),
        ];
        let tensors = vec![TensorDecl {
            name: "blk.48.nextn.eh_proj.weight".to_string(),
            dims: vec![32, 32],
            type_id: GgmlType::BF16.as_u32(),
            nbytes: 32 * 32 * 2,
        }];
        let dir = Path::new("/synthetic-qwen-folder");
        let draft = draft_header(dir, &keys, 48, tensors.clone())
            .unwrap()
            .unwrap();
        assert_eq!(draft.value("qwen4exp.block_count"), Some(&Value::U32(49)));
        assert_eq!(
            draft.value("qwen4exp.nextn_predict_layers"),
            Some(&Value::U32(1))
        );
        assert_eq!(
            draft.value("qwen4exp.attention.compress_ratios"),
            Some(&Value::Array(
                (0..48)
                    .map(|_| Value::I32(128))
                    .chain([Value::I32(0)])
                    .collect()
            ))
        );
        assert_eq!(keys[0].1, Value::U32(48));
        assert_eq!(keys[1].1, Value::Array(vec![Value::I32(128); 48]));
        assert!(draft_header(dir, &keys, 48, vec![]).unwrap().is_none());
        assert!(
            draft_header(dir, &keys, usize::MAX, tensors)
                .err()
                .expect("overflow refused")
                .to_string()
                .contains("block count exceeds u32")
        );
    }

    /// A table in F8 parts across two shards is read where the folder stores it: each part's
    /// byte range in its shard, in the parts' order, at the one tensor scale. Parts at unequal
    /// scales, a part of another width, parts short of the table's rows and a table stored in
    /// another layout (FP4-Local's NVFP4 rows) are refused by name.
    #[test]
    fn a_table_s_rows_are_read_where_the_folder_stores_them() {
        let dir = std::env::temp_dir().join(format!("folder-ple-{}", std::process::id()));
        let make = |nvfp4: bool, second_scale: f32, width: usize| {
            let hq = serde_json::json!({"quantization": {"quant_algo": "MIXED_PRECISION",
                "quantized_layers": {"e": {"quant_algo": if nvfp4 { "NVFP4" } else { "FP8" }}}}});
            std::fs::create_dir_all(&dir).unwrap();
            let (one, two) = (1.5f32.to_le_bytes(), second_scale.to_le_bytes());
            let divisor = if nvfp4 { 2 } else { 1 };
            let (a, b) = (
                vec![0x38u8; 2 * 160 / divisor],
                vec![0x40u8; 3 * width / divisor],
            );
            let (a_shape, b_shape) = ([2, 160 / divisor], [3, width / divisor]);
            let (sa_shape, sb_shape) = ([2, 160 / 16], [3, width / 16]);
            let (sa, sb) = (vec![0x38; 2 * 160 / 16], vec![0x40; 3 * width / 16]);
            let dtype = if nvfp4 { "U8" } else { "F8_E4M3" };
            let mut first: Vec<(&str, &str, &[usize], &[u8])> = vec![
                ("e.shard_0.weight", dtype, &a_shape, &a),
                (
                    if nvfp4 {
                        "e.shard_0.weight_scale_2"
                    } else {
                        "e.shard_0.weight_scale"
                    },
                    "F32",
                    &[1],
                    &one,
                ),
            ];
            let mut second: Vec<(&str, &str, &[usize], &[u8])> = vec![
                ("e.shard_1.weight", dtype, &b_shape, &b),
                (
                    if nvfp4 {
                        "e.shard_1.weight_scale_2"
                    } else {
                        "e.shard_1.weight_scale"
                    },
                    "F32",
                    &[1],
                    &two,
                ),
            ];
            if nvfp4 {
                first.push(("e.shard_0.weight_scale", "F8_E4M3", &sa_shape, &sa));
                second.push(("e.shard_1.weight_scale", "F8_E4M3", &sb_shape, &sb));
            }
            std::fs::write(dir.join("a.safetensors"), safetensors::write(&first)).unwrap();
            std::fs::write(dir.join("b.safetensors"), safetensors::write(&second)).unwrap();
            std::fs::write(
                dir.join(Folder::INDEX),
                serde_json::json!({"weight_map": first.iter().map(|t| (t.0, "a.safetensors"))
                    .chain(second.iter().map(|t| (t.0, "b.safetensors")))
                    .collect::<std::collections::BTreeMap<_, _>>()})
                .to_string(),
            )
            .unwrap();
            let folder = Folder::open(&dir).unwrap();
            let entries: Vec<(String, Dtype, Vec<usize>)> = folder
                .names()
                .map(|n| {
                    let r = folder.raw(n).unwrap();
                    (n.to_string(), r.dtype, r.shape.to_vec())
                })
                .collect();
            let p = plan(
                &Config::parse(&serde_json::json!({}), Some(&hq)).unwrap(),
                entries
                    .iter()
                    .map(|(n, d, s)| (n.as_str(), *d, s.as_slice())),
                crate::arch::module_class,
            );
            (folder, p)
        };
        let m = Mapped {
            sources: vec!["e.shard_0.weight".into(), "e.shard_1.weight".into()],
            role: Use::Rows,
            steps: vec![],
            dims: vec![160, 5],
        };
        let decl = |ty: GgmlType, rows: u64| TensorDecl {
            name: "per_layer_token_embd.weight".into(),
            dims: vec![160, rows],
            type_id: ty.as_u32(),
            nbytes: size_of(ty.as_u32(), &[160, rows]).unwrap(),
        };
        let (folder, p) = make(false, 1.5, 160);
        let got = ple_rows(&folder, &p, &decl(GgmlType::F8E4M3_ROWS, 5), &m).unwrap();
        assert_eq!((got.values.row_bytes, got.scale), (160, 1.5));
        assert!(got.block_scales.is_none());
        let read = |(path, range): &(std::path::PathBuf, std::ops::Range<u64>)| {
            std::fs::read(path).unwrap()[range.start as usize..range.end as usize].to_vec()
        };
        assert_eq!(read(&got.values.runs[0]), vec![0x38u8; 320]);
        assert_eq!(read(&got.values.runs[1]), vec![0x40u8; 480]);
        let refusal = |folder: &Folder, p: &Plan, ty, rows| {
            ple_rows(folder, p, &decl(ty, rows), &m)
                .expect_err("refused")
                .to_string()
        };
        for dims in [vec![0, 5], vec![160, 0], vec![160, u64::MAX]] {
            let mut bad = decl(GgmlType::F8E4M3_ROWS, 5);
            bad.dims = dims;
            bad.nbytes = 0;
            let e = ple_rows(&folder, &p, &bad, &m).unwrap_err().to_string();
            assert!(e.contains("not a table"), "{e}");
        }
        let e = refusal(&folder, &p, GgmlType::F8E4M3_ROWS, 6);
        assert!(e.contains("its parts hold 5 rows, the table 6"), "{e}");
        let e = refusal(&folder, &p, GgmlType::NVFP4_ROWS, 5);
        assert!(e.contains("not plain NVFP4 code and scale rows"), "{e}");
        let (folder, p) = make(false, 2.0, 160);
        let e = refusal(&folder, &p, GgmlType::F8E4M3_ROWS, 5);
        assert!(e.contains("e.shard_1.weight: tensor scale 2"), "{e}");
        let (folder, p) = make(false, 1.5, 96);
        let e = refusal(&folder, &p, GgmlType::F8E4M3_ROWS, 5);
        assert!(e.contains("rows of 96 values, the table's 160"), "{e}");
        let (folder, p) = make(true, 1.5, 160);
        let got = ple_rows(&folder, &p, &decl(GgmlType::NVFP4_ROWS, 5), &m).unwrap();
        assert_eq!((got.values.row_bytes, got.scale), (80, 1.5));
        assert_eq!(read(&got.values.runs[0]), vec![0x38; 160]);
        assert_eq!(read(&got.values.runs[1]), vec![0x40; 240]);
        let scales = got.block_scales.unwrap();
        assert_eq!(scales.row_bytes, 10);
        assert_eq!(read(&scales.runs[0]), vec![0x38; 20]);
        assert_eq!(read(&scales.runs[1]), vec![0x40; 30]);
        let (folder, p) = make(true, 2.0, 160);
        let e = refusal(&folder, &p, GgmlType::NVFP4_ROWS, 5);
        assert!(e.contains("e.shard_1.weight: tensor scale 2"), "{e}");
        let (folder, p) = make(true, 1.5, 96);
        let e = refusal(&folder, &p, GgmlType::NVFP4_ROWS, 5);
        assert!(e.contains("rows of 96 values, the table's 160"), "{e}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
