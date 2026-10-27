//! The model file a load opens, a GGUF split or a checkpoint folder, behind one owner.
//! [`ModelFile::open`] picks by what the path is: a directory is a checkpoint folder, which
//! [`folder::open`] reads by the map its config names, and anything else is the first shard of a
//! GGUF split. Either way the header, the tokenizer, the name and the weights are asked of the
//! one value, so a caller that loads a model never chooses between the two.
//!
//! [`Weights`] is the tensors' bytes by header name. Under a GGUF split it lends the mapped
//! bytes, no copy; under a folder it holds the bytes the folder's part source makes of the
//! stored planes, each in the type its header declares, so a tensor's bytes are what the
//! prepared split of the same folder carries.

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::path::{Path, PathBuf};

use gguf::{GgmlType, Split};
use model::part::{Part, Planned, Source, Take};
use tokenizer::Tokenizer;

use crate::folder;
use crate::part::Checkpoint;

/// Why a model file was refused.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The path is no GGUF split: its first shard, or a shard it names, is refused.
    #[error("open {}: {source}", .path.display())]
    Gguf {
        path: PathBuf,
        source: gguf::LoadError,
    },
    /// The directory is no checkpoint folder a map reads.
    #[error("checkpoint folder {}: {source}", .path.display())]
    Folder {
        path: PathBuf,
        source: Box<folder::Error>,
    },
    /// A GGUF split refused a tensor's read.
    #[error(transparent)]
    Split(#[from] model::part::Error),
    /// A checkpoint folder refused a tensor's read.
    #[error(transparent)]
    Checkpoint(#[from] crate::part::Error),
    /// The name is no weight of the file: a tensor it does not hold, a row table of a folder, or
    /// a scale a weight carries.
    #[error("{name}: not a weight of the model file")]
    Unknown { name: String },
    #[error(transparent)]
    Tokenizer(#[from] tokenizer::Error),
}

/// A model file as a load opens it.
pub enum ModelFile {
    /// A GGUF split.
    Gguf(Split),
    /// A checkpoint folder, opened by the map its config names.
    Folder(Box<folder::Opened>),
}

impl ModelFile {
    /// The model file at `path`: a directory is a checkpoint folder ([`folder::open`]), any other
    /// path the first shard of a GGUF split ([`Split::open`]). A refusal names the path.
    pub fn open(path: impl AsRef<Path>) -> Result<ModelFile, Error> {
        let path = path.as_ref();
        if path.is_dir() {
            let opened = folder::open(path).map_err(|source| Error::Folder {
                path: path.to_path_buf(),
                source: Box::new(source),
            })?;
            return Ok(ModelFile::Folder(Box::new(opened)));
        }
        let split = Split::open(path).map_err(|source| Error::Gguf {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(ModelFile::Gguf(split))
    }

    /// The header: a split's first shard, or the header a folder's tensors and config make, with
    /// every tensor of the file and the keys a split carries (the tokenizer's and the chat
    /// template among them).
    #[must_use]
    pub fn header(&self) -> &Split {
        match self {
            ModelFile::Gguf(split) => split,
            ModelFile::Folder(opened) => &opened.header,
        }
    }

    /// The tokenizer the header's keys build; a folder's keys were checked against Hugging
    /// Face's at [`ModelFile::open`].
    pub fn tokenizer(&self) -> Result<Tokenizer, Error> {
        Ok(Tokenizer::from_keys(self.header().keys())?)
    }

    /// The name the model goes by: the header's `general.name`, else a folder's own name; `None`
    /// for a split that states none and for a folder path with no final component.
    #[must_use]
    pub fn name(&self) -> Option<String> {
        if let Some(name) = self.header().value("general.name").and_then(|v| v.as_str()) {
            return Some(name.to_string());
        }
        let ModelFile::Folder(opened) = self else {
            return None;
        };
        // A header over no file is named by the folder; one over the folder's shards by the first.
        let at = opened.header.shard_path(0)?;
        let dir = if opened.header.is_header_only() {
            at
        } else {
            at.parent()?
        };
        dir.file_name().map(|n| n.to_string_lossy().into_owned())
    }

    /// Every tensor's bytes, planned whole from the headers alone.
    pub fn weights(&self) -> Result<Weights<'_>, Error> {
        let (backing, planned) = match self {
            ModelFile::Gguf(split) => (Backing::Split(split), split.plan(whole)?),
            ModelFile::Folder(opened) => (
                Backing::Checkpoint(&opened.source),
                opened.source.plan(whole)?,
            ),
        };
        let planned = planned
            .into_iter()
            .map(|p| (p.info.name.clone(), p))
            .collect();
        Ok(Weights { backing, planned })
    }
}

/// The rule that takes every tensor whole.
fn whole(_: &str, _: GgmlType, _: &[u64]) -> Result<Take, Infallible> {
    Ok(Take::Whole)
}

/// Where a [`Weights`] reads from.
#[derive(Clone, Copy)]
enum Backing<'a> {
    Split(&'a Split),
    Checkpoint(&'a Checkpoint),
}

/// The tensors of a [`ModelFile`], whole, by their header names: the tensors the part source
/// reads. A folder's row tables, which its header lists, and the scales a weight carries are not
/// among them.
pub struct Weights<'a> {
    backing: Backing<'a>,
    planned: BTreeMap<String, Planned>,
}

impl<'a> Weights<'a> {
    /// The names `bytes` takes, in order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.planned.keys().map(String::as_str)
    }

    /// The bytes of tensor `name`, in the type and dims its header declares. A name that is no
    /// weight of the file is refused by name.
    pub fn bytes(&self, name: &str) -> Result<Part<'a>, Error> {
        let planned = self.planned.get(name).ok_or_else(|| Error::Unknown {
            name: name.to_string(),
        })?;
        match self.backing {
            Backing::Split(split) => Ok(split.read(planned)?),
            Backing::Checkpoint(source) => Ok(source.read(planned)?),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufWriter;

    use gguf::write::{Layout, TensorDecl, Writer};

    use super::*;
    use crate::test_support::{small_qwen3_folder, temp_dir};

    /// A one-shard GGUF named `tiny` in `dir` holding the F32 tensors `t0`, `t1` and `t2` of two,
    /// three and four values, each tensor's own.
    fn tiny_gguf(dir: &Path) -> PathBuf {
        let path = dir.join("tiny.gguf");
        let tensors: Vec<(String, Vec<u8>)> = (0..3u32)
            .map(|i| {
                let bytes = (0..i + 2)
                    .flat_map(|j| ((i * 10 + j) as f32).to_le_bytes())
                    .collect();
                (format!("t{i}"), bytes)
            })
            .collect();
        let decls = tensors
            .iter()
            .map(|(name, bytes)| TensorDecl {
                name: name.clone(),
                dims: vec![bytes.len() as u64 / 4],
                type_id: 0,
                nbytes: bytes.len() as u64,
            })
            .collect();
        let name = (
            "general.name".to_string(),
            gguf::Value::String("tiny".into()),
        );
        let layout = Layout::new(&[name], decls).unwrap();
        let mut writer = Writer::new(BufWriter::new(File::create(&path).unwrap()), layout).unwrap();
        for (name, bytes) in &tensors {
            writer.tensor(name, bytes).unwrap();
        }
        writer.finish().unwrap();
        path
    }

    /// A GGUF split opens as one and its weights are its tensors by name; a tensor's bytes are the
    /// file's mapped run itself, not a copy of it.
    #[test]
    fn a_split_lends_its_mapped_bytes() {
        let dir = temp_dir("model-file-split");
        let file = ModelFile::open(tiny_gguf(&dir)).unwrap();
        assert!(matches!(file, ModelFile::Gguf(_)));
        assert_eq!(file.name().as_deref(), Some("tiny"));
        let weights = file.weights().unwrap();
        assert_eq!(weights.names().collect::<Vec<_>>(), ["t0", "t1", "t2"]);
        for name in ["t0", "t1", "t2"] {
            let part = weights.bytes(name).unwrap();
            let mapped = file.header().data(name).unwrap();
            let spans = part.spans();
            assert_eq!(spans.len(), 1, "{name}");
            assert_eq!(spans[0], mapped, "{name}");
            assert!(std::ptr::eq(spans[0], mapped), "{name}: a copy of the run");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A folder opens as one and its weights are the tensors of its header; each tensor's bytes are
    /// what the folder's part source reads of it, in the size its header declares.
    #[test]
    fn a_folder_s_weights_are_its_source_s_parts() {
        let dir = temp_dir("model-file-folder");
        small_qwen3_folder(&dir);
        let file = ModelFile::open(&dir).unwrap();
        let ModelFile::Folder(opened) = &file else {
            panic!("a directory opens as a folder");
        };
        let folder_name = dir.file_name().map(|n| n.to_string_lossy().into_owned());
        assert_eq!(file.name(), folder_name);
        let weights = file.weights().unwrap();
        let mut header: Vec<&str> = file
            .header()
            .iter_tensors()
            .map(|(_, t)| t.name.as_str())
            .collect();
        header.sort_unstable();
        assert_eq!(weights.names().collect::<Vec<_>>(), header);
        let planned = opened.source.plan(whole).unwrap();
        assert_eq!(planned.len(), header.len());
        for p in &planned {
            let name = &p.info.name;
            let bytes = weights.bytes(name).unwrap().spans().concat();
            assert_eq!(bytes.len() as u64, p.info.nbytes, "{name}");
            assert_eq!(
                bytes,
                opened.source.read(p).unwrap().spans().concat(),
                "{name}"
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A folder's tokenizer encodes as the one `open` checked against Hugging Face's.
    #[test]
    fn a_folder_s_tokenizer_is_the_checked_one() {
        let dir = temp_dir("model-file-tokenizer");
        small_qwen3_folder(&dir);
        let file = ModelFile::open(&dir).unwrap();
        let ModelFile::Folder(opened) = &file else {
            panic!("a directory opens as a folder");
        };
        let built = file.tokenizer().unwrap();
        for text in ["Aa", "aAaA", "AA"] {
            let ids = built.encode(text, true, true);
            assert!(!ids.is_empty(), "{text:?}");
            assert_eq!(ids, opened.tokenizer.encode(text, true, true), "{text:?}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A name that is no weight is refused by its name, under a split and under a folder alike.
    #[test]
    fn an_unknown_name_is_refused_by_name() {
        let dir = temp_dir("model-file-unknown");
        let folder = dir.join("folder");
        std::fs::create_dir(&folder).unwrap();
        small_qwen3_folder(&folder);
        for path in [tiny_gguf(&dir), folder] {
            let file = ModelFile::open(&path).unwrap();
            let weights = file.weights().unwrap();
            let e = weights.bytes("no.such.tensor").err().unwrap().to_string();
            assert_eq!(e, "no.such.tensor: not a weight of the model file");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// An empty directory is refused as a checkpoint folder, by its path and the config it lacks,
    /// and a path that holds no split by its path.
    #[test]
    fn a_path_that_is_no_model_file_is_refused_by_name() {
        let dir = temp_dir("model-file-refused");
        let refusal = |path: &Path| ModelFile::open(path).err().unwrap().to_string();
        let e = refusal(&dir);
        assert!(
            e.starts_with(&format!("checkpoint folder {}: ", dir.display())),
            "{e}"
        );
        assert!(
            e.contains(&format!("{}", dir.join("config.json").display())),
            "{e}"
        );
        let missing = dir.join("missing.gguf");
        let e = refusal(&missing);
        assert!(
            e.starts_with(&format!("open {}: ", missing.display())),
            "{e}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The fixture folder `<family>-mlx` and the split `prepare` made of it, `<family>-mlxsplit`,
    /// under the fixture root, read as one model file: the same weights by name, each in the
    /// same type, dims, scales and bytes, and a tokenizer that encodes a fixed text to the same
    /// ids.
    fn assert_a_folder_reads_as_its_prepared_split(family: &str) {
        let root = refset::fixture::root().unwrap();
        let first = refset::fixture::first_shard_at(
            &format!("{root}/{family}-mlxsplit"),
            "`prepare` writes it from the family's MLX fixture folder",
        )
        .unwrap();
        let folder = ModelFile::open(format!("{root}/{family}-mlx")).unwrap();
        let split = ModelFile::open(first).unwrap();
        assert!(matches!(folder, ModelFile::Folder(_)));
        assert!(matches!(split, ModelFile::Gguf(_)));
        let (a, b) = (folder.weights().unwrap(), split.weights().unwrap());
        assert_eq!(a.names().collect::<Vec<_>>(), b.names().collect::<Vec<_>>());
        for name in a.names() {
            let (x, y) = (a.bytes(name).unwrap(), b.bytes(name).unwrap());
            assert_eq!(x.info.ty, y.info.ty, "{name}");
            assert_eq!(x.info.dims, y.info.dims, "{name}");
            assert_eq!(x.scale, y.scale, "{name}");
            let (x, y) = (x.spans().concat(), y.spans().concat());
            assert!(
                x == y,
                "{name}: the bytes differ from byte {:?}",
                x.iter().zip(&y).position(|(p, q)| p != q)
            );
        }
        let text = "The quick brown fox jumps over the lazy dog. 12345\n<|im_start|>user\nhello<|im_end|>\n";
        let (x, y) = (folder.tokenizer().unwrap(), split.tokenizer().unwrap());
        assert_eq!(x.n_vocab(), y.n_vocab());
        let ids = x.encode(text, true, true);
        assert!(ids.len() > 20, "{ids:?}");
        assert_eq!(ids, y.encode(text, true, true));
    }

    #[test]
    #[ignore = "needs the fixture root's qwen3moe-mlx folder and its qwen3moe-mlxsplit split"]
    fn hw_a_qwen3_fixture_folder_reads_as_its_prepared_split() {
        assert_a_folder_reads_as_its_prepared_split("qwen3moe");
    }

    #[test]
    #[ignore = "needs the fixture root's qwen35moe-mlx folder and its qwen35moe-mlxsplit split"]
    fn hw_a_qwen35_fixture_folder_reads_as_its_prepared_split() {
        assert_a_folder_reads_as_its_prepared_split("qwen35moe");
    }
}
