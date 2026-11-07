//! Where the quantizer reads its input: one [`Source`] of tensors and header keys, which the plan and the writer read
//! through. A GGUF split is one ([`SplitSource`]); a checkpoint folder is the other (`model_hf::quantize_source`, which
//! reads the folder with the map its config names and makes each tensor's bytes as the split `prepare` writes would
//! hold them, without writing that split).

use crate::{Error, Result};
use gguf::{GENERAL_ARCHITECTURE, TensorInfo, Value};
use std::borrow::Cow;
use std::ops::Range;

/// The tensors and keys of the input set, and the bytes of any run of a tensor.
pub trait Source: Sync {
    /// What a log line names the input as.
    fn label(&self) -> String;

    /// The first shard's keys, in order, with the `split.*` keys of a split input among them.
    fn keys(&self) -> &[(String, Value)];

    /// The input's shards: `--keep-split` writes one output shard each.
    fn shard_count(&self) -> usize;

    /// Every tensor with its shard; within a shard `offset` rises in file order.
    fn tensors(&self) -> Vec<(usize, TensorInfo)>;

    /// Bytes `bytes` of tensor `name`'s data (whole rows of the tensor, in its stored type); a name or a range the
    /// input does not hold is refused by name.
    fn read(&self, name: &str, bytes: Range<u64>) -> Result<Cow<'_, [u8]>>;

    /// The value of `key`, if the header has it.
    fn value(&self, key: &str) -> Option<&Value> {
        self.keys().iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// `general.architecture`.
    fn architecture(&self) -> Option<&str> {
        self.value(GENERAL_ARCHITECTURE).and_then(Value::as_str)
    }

    /// `<architecture>.<suffix>`, the key a reader of the header names in a refusal.
    fn arch_key(&self, suffix: &str) -> String {
        match self.architecture() {
            Some(arch) => format!("{arch}.{suffix}"),
            None => suffix.to_string(),
        }
    }

    /// `<architecture>.<suffix>` as an unsigned integer.
    fn arch_get_u64(&self, suffix: &str) -> Option<u64> {
        self.architecture()?;
        self.value(&self.arch_key(suffix)).and_then(Value::as_u64)
    }
}

/// A GGUF split read in place: the bytes are the mapped file's, no copy.
pub struct SplitSource(pub gguf::Split);

impl Source for SplitSource {
    fn label(&self) -> String {
        self.0
            .shard_path(0)
            .map_or_else(|| "the input set".to_string(), |p| p.display().to_string())
    }

    fn keys(&self) -> &[(String, Value)] {
        self.0.keys()
    }

    fn shard_count(&self) -> usize {
        self.0.shard_count()
    }

    fn tensors(&self) -> Vec<(usize, TensorInfo)> {
        self.0
            .iter_tensors()
            .map(|(shard, t)| (shard, t.clone()))
            .collect()
    }

    fn read(&self, name: &str, bytes: Range<u64>) -> Result<Cow<'_, [u8]>> {
        let data = self
            .0
            .data(name)
            .map_err(|e| Error::refused(format!("{name}: {e}")))?;
        usize::try_from(bytes.start)
            .ok()
            .zip(usize::try_from(bytes.end).ok())
            .and_then(|(a, b)| data.get(a..b))
            .map(Cow::Borrowed)
            .ok_or_else(|| {
                Error::refused(format!(
                    "{name}: bytes {}..{} of a tensor of {}",
                    bytes.start,
                    bytes.end,
                    data.len()
                ))
            })
    }
}
