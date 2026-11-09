//! The metadata llama-quantize writes (`llama_model_quantize_impl`, `tools/quantize/quantize.cpp`), key by key and in
//! its order: the first input shard's pairs without its `split.*` keys, `general.quantization_version` and
//! `general.file_type` and `quantize.imatrix.*` each removed where it was and appended, and for a split output every
//! shard's `split.no`, `split.count` and `split.tensors.count` appended. The first shard carries all of it; the
//! others carry only their three `split.*` pairs, as llama-quantize's `gguf_init_empty` contexts do.

use crate::cli::Args;
use crate::imatrix::Imatrix;
use crate::source::Source;
use gguf::Value;

/// `GGML_QNT_VERSION`.
const QNT_VERSION: u32 = 2;
const SPLIT_KEYS: [&str; 3] = ["split.no", "split.count", "split.tensors.count"];

/// `gguf_set_val_*`: remove the key's pair if there is one, then append, so a key set again moves to the end.
fn put(kvs: &mut Vec<(String, Value)>, key: &str, v: Value) {
    kvs.retain(|(k, _)| k != key);
    kvs.push((key.to_string(), v));
}

/// `strncpy(…, s, 127)` into a 128-byte `val_str`: the first 127 bytes, as text.
fn val_str(s: &str) -> String {
    let b = s.as_bytes();
    String::from_utf8_lossy(&b[..b.len().min(127)]).into_owned()
}

/// The metadata of every output shard: `n_split` lists, the first the whole of it.
#[must_use]
pub fn output_kvs(
    set: &dyn Source,
    a: &Args,
    im: Option<&Imatrix>,
    n_tensors: usize,
    n_split: usize,
) -> Vec<Vec<(String, Value)>> {
    let mut kvs: Vec<(String, Value)> = set.keys().to_vec();
    put(
        &mut kvs,
        "general.quantization_version",
        Value::U32(QNT_VERSION),
    );
    put(&mut kvs, "general.file_type", Value::U32(a.ftype.id()));
    kvs.retain(|(k, _)| !SPLIT_KEYS.contains(&k.as_str()));
    if let Some(im) = im {
        put(
            &mut kvs,
            "quantize.imatrix.file",
            Value::String(val_str(&im.path)),
        );
        if let Some(d) = im.datasets.first() {
            put(
                &mut kvs,
                "quantize.imatrix.dataset",
                Value::String(val_str(d)),
            );
        }
        put(
            &mut kvs,
            "quantize.imatrix.entries_count",
            Value::U32(im.entry_count() as u32),
        );
        if im.chunk_count > 0 {
            put(
                &mut kvs,
                "quantize.imatrix.chunks_count",
                Value::U32(im.chunk_count),
            );
        }
    }
    let mut out = Vec::with_capacity(n_split);
    for i in 0..n_split {
        let mut kv = if i == 0 { kvs.clone() } else { Vec::new() };
        if n_split > 1 {
            put(&mut kv, "split.no", Value::U16(i as u16));
            put(&mut kv, "split.count", Value::U16(n_split as u16));
            put(&mut kv, "split.tensors.count", Value::I32(n_tensors as i32));
        }
        out.push(kv);
    }
    out
}
