//! The model file a gate or a server opens, named on its command line: `-m PATH` (or another
//! spelling the binary gives the path flag) or, in a build with the `hf` feature, `--hf
//! <repo>[:<quant>]`, fetched into the cache ([`fetch`]). A server takes the flags out of its
//! arguments (`take`) before anything else parses them, and the path it resolves is the process's
//! model file ([`set_model_file`]), which [`crate::ref_model_path`] reads before the gates'
//! `$BLOOMERY_REF_MODEL`. The `hf` feature adds what fetches: the rules of a model named twice
//! (`hf::source`'s), a decision model's head from the repo a `--hf` repo's model card says it
//! quantizes (`quantized_from`), a family's MTP draft fetched beside a set only when its
//! architecture's rule says the set can run that file, and a seat's image projector under `--hf`
//! (`hf_projector`), the repo's own beside the model. A `--hf` repo that holds a safetensors
//! checkpoint and no GGUF set resolves to the checkpoint's folder only for a caller that can
//! open checkpoints; other callers refuse it before fetching its files.

use std::path::{Path, PathBuf};
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
pub const NONE: &str = "no model file: pass -m PATH (the first shard, or a checkpoint folder) \
                               or --hf <repo>[:<quant>], or set BLOOMERY_REF_MODEL";

#[cfg(feature = "checkpoint")]
static KEPT: std::sync::Mutex<Option<(PathBuf, model_hf::folder::Opened)>> =
    std::sync::Mutex::new(None);

/// The folder at `path`, reusing the open whose header selected its server seat.
#[cfg(feature = "checkpoint")]
pub fn open_folder(path: &Path) -> Result<model_hf::folder::Opened, GateError> {
    let kept = KEPT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take();
    match kept {
        Some((p, opened)) if p == path => Ok(opened),
        _ => read_folder(path),
    }
}

#[cfg(feature = "checkpoint")]
fn read_folder(path: &Path) -> Result<model_hf::folder::Opened, GateError> {
    model_hf::folder::open(path)
        .map_err(|e| format!("checkpoint folder {}: {e}", path.display()).into())
}

/// `f` of the model's header, keeping a checkpoint folder for its seat's open.
pub fn with_header<R>(path: &Path, f: impl FnOnce(&Split) -> R) -> Result<R, GateError> {
    #[cfg(feature = "checkpoint")]
    if path.is_dir() {
        let mut kept = KEPT
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let opened = match kept.take() {
            Some((p, opened)) if p == path => opened,
            _ => read_folder(path)?,
        };
        let result = f(&opened.header);
        *kept = Some((path.to_path_buf(), opened));
        return Ok(result);
    }
    header(path).map(|header| f(&header))
}

/// The header of the model at `path`, which a server seats it by and `-ngl N` is compared with:
/// its GGUF split's first shard, or a checkpoint folder's when `path` is a directory, which the
/// map its config names reads (`model_hf::folder::open`), so the seat is chosen from the header's
/// `general.architecture` as a split's is.
pub fn header(path: &Path) -> Result<Split, GateError> {
    if path.is_dir() {
        return folder_header(path);
    }
    Split::open(path).map_err(|e| format!("open {}: {e}", path.display()).into())
}

/// The header of the checkpoint folder `path`.
#[cfg(feature = "checkpoint")]
fn folder_header(path: &Path) -> Result<Split, GateError> {
    read_folder(path).map(|f| f.header)
}

/// A checkpoint folder in a build without the `checkpoint` feature, refused by name.
#[cfg(not(feature = "checkpoint"))]
fn folder_header(path: &Path) -> Result<Split, GateError> {
    Err(format!(
        "{} is a directory: a checkpoint folder is opened by a build with the checkpoint feature",
        path.display()
    )
    .into())
}

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

/// The refusal a generative seat gives a file that declares a decision type: a decision model is
/// the decide seat's, whatever trunk it shares with a generative family. A marker that is not a
/// string is `decision_type`'s error.
pub fn refuse_decision_model(file: &Split) -> Result<(), GateError> {
    match decision_type(file)? {
        None => Ok(()),
        Some(kind) => Err(format!(
            "a {} file declares {} = {kind}: a decision model is the decide seat's, not generation",
            file.architecture().unwrap_or("<missing>"),
            file.arch_key("decision.type")
        )
        .into()),
    }
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

/// A header of architecture `arch` with `keys` beside it, written to `path` and opened: the files
/// the decision-marker tests read.
#[cfg(test)]
pub(crate) fn test_header(
    path: &std::path::Path,
    arch: &str,
    keys: Vec<(String, gguf::Value)>,
) -> Split {
    let mut all = vec![(
        gguf::GENERAL_ARCHITECTURE.to_owned(),
        gguf::Value::String(arch.into()),
    )];
    all.extend(keys);
    let file = std::fs::File::create_new(path).unwrap();
    let layout = gguf::write::Layout::new(&all, Vec::new()).unwrap();
    gguf::write::Writer::new(file, layout)
        .and_then(gguf::write::Writer::finish)
        .unwrap();
    Split::open(path).unwrap()
}

#[cfg(test)]
mod tests {
    use super::{Split, decision_type, header, refuse_decision_model, test_header};
    use gguf::Value;

    /// A header of architecture `fixture` with the optional `fixture.decision.type` marker.
    fn marked_header(path: &std::path::Path, marker: Option<Value>) -> Split {
        let keys = marker
            .map(|m| vec![("fixture.decision.type".to_owned(), m)])
            .unwrap_or_default();
        test_header(path, "fixture", keys)
    }

    /// A path that is no model file is refused naming it, and a directory is a checkpoint folder:
    /// a build without the `checkpoint` feature refuses it by that name instead of failing to
    /// open it as a GGUF file, which `-ngl N` read as the file's layer count before.
    #[test]
    fn a_header_is_a_file_or_a_checkpoint_folder() {
        let e = match header(std::path::Path::new("/no/such/model.gguf")) {
            Ok(_) => panic!("no such file opened"),
            Err(e) => e.to_string(),
        };
        assert!(e.starts_with("open /no/such/model.gguf: "), "{e}");
        #[cfg(not(feature = "checkpoint"))]
        {
            let dir = std::env::temp_dir();
            let e = match header(&dir) {
                Ok(_) => panic!("a directory opened as a model file"),
                Err(e) => e.to_string(),
            };
            assert_eq!(
                e,
                format!(
                    "{} is a directory: a checkpoint folder is opened by a build with the \
                     checkpoint feature",
                    dir.display()
                )
            );
        }
    }

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
            let path = std::env::temp_dir().join(format!(
                "bloomery-decision-header-{}-{index}.gguf",
                std::process::id()
            ));
            let split = marked_header(&path, marker);
            match expected {
                Ok(value) => assert_eq!(decision_type(&split).unwrap(), value),
                Err(error) => assert_eq!(decision_type(&split).unwrap_err().to_string(), error),
            }
            drop(split);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn a_declared_decision_type_is_refused_by_name_and_no_marker_passes() {
        let cases = [
            (None, Ok(())),
            (
                Some(Value::String("lev".into())),
                Err(
                    "a fixture file declares fixture.decision.type = lev: a decision model is the decide seat's, not generation",
                ),
            ),
            (
                Some(Value::U32(1)),
                Err("fixture.decision.type: expected a string"),
            ),
        ];
        for (index, (marker, expected)) in cases.into_iter().enumerate() {
            let path = std::env::temp_dir().join(format!(
                "bloomery-decision-refuse-{}-{index}.gguf",
                std::process::id()
            ));
            let split = marked_header(&path, marker);
            let got = refuse_decision_model(&split).map_err(|e| e.to_string());
            assert_eq!(got, expected.map_err(str::to_owned));
            drop(split);
            std::fs::remove_file(path).unwrap();
        }
    }
}
