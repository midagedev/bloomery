//! Inventory contracts, on synthetic files: the header-only path carries
//! what the strict reader refuses, reports unknown sizes as `None`, and
//! agrees with `Gguf::open` on every field for types both accept.

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
