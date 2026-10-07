//! Inventory contracts, on synthetic files: the header-only path carries
//! what the strict reader refuses, reports unknown sizes as `None`, agrees
//! with `Gguf::open` on every field for types both accept, and refuses, as
//! it does, metadata ggml does not read (an array of arrays, an alignment
//! that is 0 or not a u32).

use std::path::PathBuf;

use gguf::{GgmlType, Gguf, LoadError, inventory_of};

fn put_u32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn put_u64(v: &mut Vec<u8>, x: u64) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn put_str(v: &mut Vec<u8>, s: &str) {
    put_u64(v, s.len() as u64);
    v.extend_from_slice(s.as_bytes());
}

/// A minimal GGUF v3: magic, version, one `general.alignment` KV, the given
/// tensor infos, then `data_len` zero bytes past the aligned data base —
/// enough for the strict reader to reach its own refusal or accept.
fn write_gguf(path: &std::path::Path, tensors: &[(&str, &[u64], u32)], data_len: usize) {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    put_u32(&mut b, 3);
    put_u64(&mut b, tensors.len() as u64);
    put_u64(&mut b, 1);
    put_str(&mut b, "general.alignment");
    put_u32(&mut b, 4); // u32
    put_u32(&mut b, 32);
    for (name, dims, ty) in tensors {
        put_str(&mut b, name);
        put_u32(&mut b, dims.len() as u32);
        for d in *dims {
            put_u64(&mut b, *d);
        }
        put_u32(&mut b, *ty);
        put_u64(&mut b, 0);
    }
    while b.len() % 32 != 0 {
        b.push(0);
    }
    b.resize(b.len() + data_len, 0);
    std::fs::write(path, b).expect("write synthetic gguf");
}

fn tmp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("gguf-inventory-{}-{name}", std::process::id()));
    p
}

/// A file of no tensors whose one metadata pair is `key`, of value tag
/// `tag`, with the value's bytes `value`.
fn one_kv(path: &std::path::Path, key: &str, tag: u32, value: &[u8]) {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    put_u32(&mut b, 3);
    put_u64(&mut b, 0);
    put_u64(&mut b, 1);
    put_str(&mut b, key);
    put_u32(&mut b, tag);
    b.extend_from_slice(value);
    std::fs::write(path, b).expect("write synthetic gguf");
}

/// Both entry points' verdict on the file at `path`.
fn both(path: &std::path::Path) -> [Result<(), LoadError>; 2] {
    [inventory_of(path).map(|_| ()), Gguf::open(path).map(|_| ())]
}

/// An array holds one scalar or string type, as ggml's reader requires: both
/// entry points refuse an array of arrays by its key, and an element tag that
/// is no value type even when the array is empty.
#[test]
fn an_array_holds_no_arrays() {
    let p = tmp("nested.gguf");
    // Element tag 9, one element: itself an empty array of u32.
    let mut nested = Vec::new();
    put_u32(&mut nested, 9);
    put_u64(&mut nested, 1);
    put_u32(&mut nested, 4);
    put_u64(&mut nested, 0);
    one_kv(&p, "k", 9, &nested);
    for r in both(&p) {
        match r {
            Err(LoadError::NestedArray { key }) => assert_eq!(key, "k"),
            other => panic!("an array of arrays must be refused by name, got {other:?}"),
        }
    }
    // Element tag 13, no elements.
    let mut empty = Vec::new();
    put_u32(&mut empty, 13);
    put_u64(&mut empty, 0);
    one_kv(&p, "k", 9, &empty);
    for r in both(&p) {
        match r {
            Err(LoadError::BadValueType(13)) => {}
            other => panic!("element tag 13 must be refused, got {other:?}"),
        }
    }
    std::fs::remove_file(&p).unwrap();
}

/// `general.alignment` is a u32, as ggml reads it, and not 0: both entry
/// points refuse a 0 and a u16 by name instead of dividing by zero or taking
/// another type's value.
#[test]
fn the_alignment_is_a_nonzero_u32() {
    let p = tmp("alignment.gguf");
    for (tag, value, shown) in [
        (4, 0u32.to_le_bytes().to_vec(), "U32(0)"),
        (2, 32u16.to_le_bytes().to_vec(), "U16(32)"),
    ] {
        one_kv(&p, "general.alignment", tag, &value);
        for r in both(&p) {
            match r {
                Err(LoadError::Alignment { value }) => assert_eq!(value, shown),
                other => panic!("alignment {shown} must be refused by name, got {other:?}"),
            }
        }
    }
    std::fs::remove_file(&p).unwrap();
}

/// The contract this crate exists for: a tensor whose ggml type the engine
/// has no size for (q4_0, id 2 — ggml.h:394) inventories with a size from
/// ggml's table while `Gguf::open` refuses the file outright.
#[test]
fn inventory_carries_what_the_strict_reader_refuses() {
    let p = tmp("q4_0.gguf");
    // dims 32x4: one 18-byte block per row, four rows.
    write_gguf(&p, &[("blk.0.attn_q.weight", &[32, 4], 2)], 72);

    let inv = inventory_of(&p).expect("inventory");
    assert_eq!(inv.version, 3);
    assert_eq!(inv.tensors.len(), 1);
    let t = &inv.tensors[0];
    assert_eq!(t.name, "blk.0.attn_q.weight");
    assert_eq!(t.dims, vec![32, 4]);
    assert_eq!(t.type_id, 2);
    assert_eq!(t.nbytes, Some(72));

    match Gguf::open(&p) {
        Err(LoadError::UnsupportedType { ty, .. }) => assert_eq!(ty, GgmlType::Unknown(2)),
        Err(other) => panic!("strict open refused with the wrong error: {other}"),
        Ok(_) => panic!("strict open must refuse q4_0"),
    }
}

/// A type id outside both the engine's table and ggml's (41, Bonsai
/// Q1_0_G128) still inventories — with `None` bytes, never a rejection.
#[test]
fn inventory_reports_none_for_types_outside_the_table() {
    let p = tmp("unknown.gguf");
    write_gguf(&p, &[("blk.0.attn_v.weight", &[256, 8], 41)], 8);

    let inv = inventory_of(&p).expect("inventory");
    assert_eq!(inv.tensors.len(), 1);
    assert_eq!(inv.tensors[0].type_id, 41);
    assert_eq!(inv.tensors[0].nbytes, None);
}

/// Both paths share one parser, so a type both accept (Q5_0) yields the
/// same name, dims, type, offset and byte count from either entry point.
#[test]
fn inventory_agrees_with_the_strict_reader_on_supported_types() {
    let p = tmp("q5_0.gguf");
    // dims 64x3: two 22-byte blocks per row, three rows = 132.
    write_gguf(&p, &[("blk.0.ffn_down.weight", &[64, 3], 6)], 132);

    let inv = inventory_of(&p).expect("inventory");
    let g = Gguf::open(&p).expect("strict open of a supported type");
    assert_eq!(inv.tensors.len(), g.tensor_count());
    let t = &inv.tensors[0];
    let s = g.tensor(0).unwrap();
    assert_eq!(t.name, s.name);
    assert_eq!(t.dims, s.dims);
    assert_eq!(t.type_id, s.ty.as_u32());
    assert_eq!(t.offset, s.offset);
    assert_eq!(t.nbytes, Some(s.nbytes));
    assert_eq!(t.nbytes, Some(132));
}
