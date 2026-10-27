//! A Qwen3.8 or Qwen3.6 ([`super::Family`]) checkpoint folder in the names `prepare` writes its
//! split by: the common reader ([`crate::folder`]) instantiated on [`Checkpoint`], the qwen35moe
//! map, and its MTP layer as the draft reads it ([`mtp`]).

use std::collections::HashMap;
use std::path::Path;

use gguf::write::{Layout, TensorDecl};
use gguf::{Split, Value};
use hf::quant_config::{Config, Plan};
use safetensors::{Dtype, Folder};

pub use crate::folder::{Error, served_as_hf};

use super::{Checkpoint, Step};

/// One tensor of the split: its declaration and how its bytes are made.
pub type Planned = crate::folder::Planned<Step>;

/// How a planned tensor's bytes are made.
pub type Make = crate::folder::Make<Step>;

/// [`crate::folder::plan_tensors`] of `entries` as qwen4exp's map reads them: the plan and the
/// planned tensors.
pub fn plan_tensors(
    checkpoint: &Checkpoint,
    config: &Config,
    entries: &[(String, Dtype, Vec<usize>)],
    q8: bool,
) -> Result<(Plan, Vec<Planned>), Error> {
    let planning = crate::folder::plan_tensors(checkpoint, config, entries, q8)?;
    Ok((planning.plan, planning.planned))
}

/// The MTP layer of the folder as the draft reads it ([`crate::folder::FolderMap::draft`]),
/// every tensor in its stored format: its header ([`draft_header`] over the target's keys `kvs`,
/// named by `dir` as the target's header is) and its planned tensors, `None` for a folder that
/// holds no `mtp.` tensor. A layer the map or the planner refuses (a second MTP layer, an F32
/// projection, an unquantized expert stack) is refused by name.
pub(super) fn mtp(
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
    // The draft's tensors are all made of their sources, none kept as stored, so no stored
    // dtype is looked up.
    let tensors = crate::folder::planned_tensors(
        plan,
        super::map(
            &checkpoint.geometry,
            logical.iter().map(|(n, s)| (n.as_str(), s.as_slice())),
            true,
        )?,
        false,
        &HashMap::new(),
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use gguf::GgmlType;
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
            crate::folder::open(&dir)
        };
        let target = [names::token_embd(), names::output()];
        let held = |o: &crate::folder::Opened| {
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
}
