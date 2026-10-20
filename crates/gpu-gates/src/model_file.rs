//! The model file a gate or a server opens, named on its command line: `-m PATH` (or another
//! spelling the binary gives the path flag) or, in a build with the `hf` feature, `--hf
//! <repo>[:<quant>]`, fetched into the cache ([`fetch`]). A server takes the flags out of its
//! arguments (`take`) before anything else parses them, and the path it resolves is the process's
//! model file ([`set_model_file`]), which [`crate::ref_model_path`] reads before the gates'
//! `$BLOOMERY_REF_MODEL`. The `hf` feature adds what fetches: the rules of a model named twice
//! (`hf::source`'s), a decision model's head from the repo a `--hf` repo's model card says it
//! quantizes (`quantized_from`), a family's MTP draft fetched beside a set only when its
//! architecture's rule says the set can run that file, and a seat's image projector under `--hf`
//! (`hf_projector`), the repo's own beside the model.

use std::path::PathBuf;
use std::sync::OnceLock;

use gguf::Split;

use crate::GateError;

#[cfg(feature = "hf")]
mod fetch;
#[cfg(feature = "hf")]
pub use fetch::{
    fetch_exact, hf_projector, hf_repo, name, quantized_from, resolve, sentence_prompts, take,
};

/// The refusal when nothing names the model.
pub(crate) const NONE: &str = "no model file: pass -m PATH (the first shard) or --hf <repo>[:<quant>], \
                               or set BLOOMERY_REF_MODEL";

/// A present decision marker is a string; malformed metadata is not an absent head.
pub fn decision_type(file: &Split) -> Result<Option<&str>, GateError> {
    let key = file.arch_key("decision.type");
    file.value(&key)
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| format!("{key}: expected a string").into())
        })
        .transpose()
}

/// The model file a server named on its command line, set once.
static MODEL_FILE: OnceLock<PathBuf> = OnceLock::new();

/// The process's model file is `path` from now on; a second call is
/// refused by name.
pub fn set_model_file(path: PathBuf) -> Result<(), GateError> {
    let shown = path.display().to_string();
    MODEL_FILE.set(path).map_err(|_| {
        format!(
            "the model file is set already ({}), and a second set names {shown}",
            MODEL_FILE
                .get()
                .map_or_else(String::new, |p| p.display().to_string())
        )
        .into()
    })
}

/// The model file a server set, if one.
pub(crate) fn model_file() -> Option<&'static PathBuf> {
    MODEL_FILE.get()
}

/// The `--hf` repo the process's model file came from; no build without the `hf` feature has one.
#[cfg(not(feature = "hf"))]
#[must_use]
pub fn hf_repo() -> Option<&'static str> {
    None
}

/// The image projector the `--hf` repo holds beside the process's model file; `None` in a build
/// without the `hf` feature, whose model file is always named by path.
#[cfg(not(feature = "hf"))]
pub fn hf_projector() -> Result<Option<PathBuf>, GateError> {
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::{Split, decision_type};
    use gguf::Value;

    #[test]
    fn decision_header_distinguishes_absent_string_and_malformed_markers() {
        let cases = [
            (None, Ok(None)),
            (Some(Value::String("clef".into())), Ok(Some("clef"))),
            (Some(Value::String(String::new())), Ok(Some(""))),
            (
                Some(Value::U32(1)),
                Err("fixture.decision.type: expected a string"),
            ),
        ];
        for (index, (marker, expected)) in cases.into_iter().enumerate() {
            let mut keys = vec![(
                gguf::GENERAL_ARCHITECTURE.to_owned(),
                Value::String("fixture".into()),
            )];
            if let Some(marker) = marker {
                keys.push(("fixture.decision.type".into(), marker));
            }
            let path = std::env::temp_dir().join(format!(
                "bloomery-decision-header-{}-{index}.gguf",
                std::process::id()
            ));
            let file = std::fs::File::create_new(&path).unwrap();
            let layout = gguf::write::Layout::new(&keys, Vec::new()).unwrap();
            gguf::write::Writer::new(file, layout)
                .and_then(gguf::write::Writer::finish)
                .unwrap();
            let split = Split::open(&path).unwrap();
            match expected {
                Ok(value) => assert_eq!(decision_type(&split).unwrap(), value),
                Err(error) => assert_eq!(decision_type(&split).unwrap_err().to_string(), error),
            }
            drop(split);
            std::fs::remove_file(path).unwrap();
        }
    }
}
