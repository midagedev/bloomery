//! A checked-in header — a key table in JSON, a tensor table in TSV — laid out as a header-only
//! `Split`: the keys, shapes and types of a real file and none of a tensor's bytes, read through
//! the engine's own `Split`.
//!
//! The key table maps each key to its value; a tokenizer array the fetch summarized is an object
//! (`{array_len, head}`) and is left out of the `Split`. A tensor line is `name`, type name,
//! dims joined by `x`, and the shard.

use gguf::write::{Layout, TensorDecl};
use gguf::{GgmlType, Split, Value};

/// The header of the tables `kv` and `tensors` as a `Split` holding no tensor's bytes, laid out at
/// a path of its own under `tag` (the tests run in threads of one process).
pub fn header_of(tag: &str, kv: &str, tensors: &str) -> Split {
    let table =
        serde_json::from_str::<serde_json::Value>(kv).expect("the checked-in header table parses");
    let table = table.as_object().expect("the header table is an object");
    let kvs: Vec<(String, Value)> = table
        .iter()
        .filter_map(|(k, v)| value(k, v).map(|v| (k.clone(), v)))
        .collect();
    let tensors: Vec<TensorDecl> = tensors.lines().map(decl).collect();
    let layout = Layout::new(&kvs, tensors).expect("the header lays out");
    let path = std::env::temp_dir().join(format!("bloomery-hdr-{tag}-{}.gguf", std::process::id()));
    Split::header_only(path, &layout).expect("the header-only split opens")
}

/// One line of a tensor table as the declaration the header lays out: its bytes the file's, whole
/// blocks of every row.
fn decl(line: &str) -> TensorDecl {
    let mut fields = line.split('\t');
    let (Some(name), Some(ty), Some(dims)) = (fields.next(), fields.next(), fields.next()) else {
        panic!("{line}: not name, type, dims, shard");
    };
    let dims: Vec<u64> = dims.split('x').map(|d| d.parse().expect("a dim")).collect();
    let ty = ty_of(ty);
    let rows = dims[1..].iter().product::<u64>();
    let per_row = ty.type_size().expect("a size") * (dims[0] / ty.blck_size().expect("a block"));
    TensorDecl {
        name: name.to_string(),
        dims,
        type_id: ty.as_u32(),
        nbytes: per_row * rows,
    }
}

/// A header value as the GGUF's: ints as U32 or U64, floats as F32, strings, bools and arrays of
/// them; `None` for a summarized tokenizer table.
fn value(key: &str, v: &serde_json::Value) -> Option<Value> {
    use serde_json::Value as J;
    match v {
        J::Bool(b) => Some(Value::Bool(*b)),
        J::String(s) => Some(Value::String(s.clone())),
        J::Number(n) => {
            if let Some(i) = n.as_u64() {
                Some(if i <= u64::from(u32::MAX) {
                    Value::U32(i as u32)
                } else {
                    Value::U64(i)
                })
            } else if let Some(i) = n.as_i64() {
                u64::try_from(i).ok().map(Value::U64)
            } else {
                n.as_f64().map(|f| Value::F32(f as f32))
            }
        }
        J::Array(items) => Some(Value::Array(
            items
                .iter()
                .map(|item| value(key, item))
                .collect::<Option<_>>()?,
        )),
        J::Object(_) => None,
        J::Null => panic!("{key}: the header table holds a null"),
    }
}

/// A ggml type by its table's name (the table spells the ids in capitals, `ggml_type_name` in
/// mixed case).
fn ty_of(name: &str) -> GgmlType {
    (0..=45)
        .chain(2000..=2005)
        .map(GgmlType::from_u32)
        .find(|ty| ty.name().is_some_and(|n| n.eq_ignore_ascii_case(name)))
        .unwrap_or_else(|| panic!("{name}: no ggml type of this name"))
}
