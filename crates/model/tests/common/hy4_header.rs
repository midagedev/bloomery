//! The checked-in Hy4-flash header (`hy4_hdr.kv.json`, `hy4_tensors.tsv`) as a header-only `Split`, with
//! the token map the key table summarizes given the file's measured shape: what `hy4_meta` and
//! `hy4_fixture` read the real file's header through.

use gguf::Split;

#[path = "header_tables.rs"]
pub mod header_tables;

/// The checked-in header tables.
pub const KV: &str = include_str!("../hy4_hdr.kv.json");
pub const TENSORS: &str = include_str!("../hy4_tensors.tsv");

pub const VOCAB: usize = 120_832;
/// The token map's compressed range: ids below [`IDENTITY_FROM`] map into `0..COMPRESSED`.
pub const COMPRESSED: usize = 92_620;
/// The first of the last 14 ids, whose map entry is the id itself.
pub const IDENTITY_FROM: usize = 120_818;
/// `engram.pad_token_id`.
pub const PAD: usize = 120_002;
/// The map's value at [`PAD`]: the id the pad slot hashes as.
pub const PAD_MAPPED: u64 = 91_804;

/// The map `engram.token_map` is given: the file's measured shape. Every id below
/// [`IDENTITY_FROM`] maps into `0..COMPRESSED` (`t · COMPRESSED / IDENTITY_FROM`), the pad maps to
/// [`PAD_MAPPED`] as the file's does, and the last 14 ids map to themselves.
pub fn token_map() -> Vec<u64> {
    (0..VOCAB)
        .map(|t| {
            if t == PAD {
                PAD_MAPPED
            } else if t >= IDENTITY_FROM {
                t as u64
            } else {
                (t * COMPRESSED / IDENTITY_FROM) as u64
            }
        })
        .collect()
}

/// The key table with `engram.token_map` as the array [`token_map`] makes, and `edit` applied to
/// the object.
pub fn kv_with(edit: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>)) -> String {
    let mut table: serde_json::Value = serde_json::from_str(KV).expect("the key table parses");
    let object = table.as_object_mut().expect("the key table is an object");
    object.insert(
        "hy_v4_flash.engram.token_map".to_string(),
        serde_json::Value::Array(token_map().into_iter().map(Into::into).collect()),
    );
    edit(object);
    table.to_string()
}

/// The header of the checked-in tables as a `Split`, `tag` its file's name.
pub fn header(tag: &str) -> Split {
    header_tables::header_of(tag, &kv_with(|_| {}), TENSORS)
}
