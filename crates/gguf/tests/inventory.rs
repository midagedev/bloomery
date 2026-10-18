//! Inventory contracts, on synthetic files: the header-only path carries
//! what the strict reader refuses, reports unknown sizes as `None`, agrees
//! with `Gguf::open` on every field for types both accept, and refuses, as
//! it does, metadata ggml does not read (an array of arrays, an alignment
//! that is 0 or not a u32) and a tensor whose byte count passes u64. The
//! same synthetic files hold `Gguf::data` to the mapping's bounds, and
//! `gguf-inventory` to a top-k within the expert count.

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

/// A minimal GGUF v3: magic, version, the given metadata pairs (key, value tag, value
/// bytes), the given tensor infos, then `data_len` zero bytes past the aligned data base —
/// enough for the strict reader to reach its own refusal or accept.
fn write_gguf_kvs(
    path: &std::path::Path,
    kvs: &[(&str, u32, Vec<u8>)],
    tensors: &[(&str, &[u64], u32)],
    data_len: usize,
) {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    put_u32(&mut b, 3);
    put_u64(&mut b, tensors.len() as u64);
    put_u64(&mut b, kvs.len() as u64);
    for (key, tag, value) in kvs {
        put_str(&mut b, key);
        put_u32(&mut b, *tag);
        b.extend_from_slice(value);
    }
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

/// [`write_gguf_kvs`] with `general.alignment` 32 as the one metadata pair.
fn write_gguf(path: &std::path::Path, tensors: &[(&str, &[u64], u32)], data_len: usize) {
    let kvs = [("general.alignment", 4, 32u32.to_le_bytes().to_vec())];
    write_gguf_kvs(path, &kvs, tensors, data_len);
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

/// The contract this crate exists for: a tensor whose type the strict reader
/// refuses (q8_K, id 15, an activation form no weight is stored in)
/// inventories with a size from the type table while `Gguf::open` refuses the
/// file outright.
#[test]
fn inventory_carries_what_the_strict_reader_refuses() {
    let p = tmp("q8_k.gguf");
    // dims 256x4: one 292-byte block per row, four rows.
    write_gguf(&p, &[("blk.0.attn_q.weight", &[256, 4], 15)], 1168);

    let inv = inventory_of(&p).expect("inventory");
    assert_eq!(inv.version, 3);
    assert_eq!(inv.tensors.len(), 1);
    let t = &inv.tensors[0];
    assert_eq!(t.name, "blk.0.attn_q.weight");
    assert_eq!(t.dims, vec![256, 4]);
    assert_eq!(t.type_id, 15);
    assert_eq!(t.nbytes, Some(1168));

    match Gguf::open(&p) {
        Err(LoadError::UnsupportedType { ty, .. }) => assert_eq!(ty, GgmlType::Q8_K),
        Err(other) => panic!("strict open refused with the wrong error: {other}"),
        Ok(_) => panic!("strict open must refuse q8_K"),
    }
}

/// A type id outside both the engine's table and ggml's (99, ik's q8_2_x4)
/// still inventories — with `None` bytes, never a rejection.
#[test]
fn inventory_reports_none_for_types_outside_the_table() {
    let p = tmp("unknown.gguf");
    write_gguf(&p, &[("blk.0.attn_v.weight", &[256, 8], 99)], 8);

    let inv = inventory_of(&p).expect("inventory");
    assert_eq!(inv.tensors.len(), 1);
    assert_eq!(inv.tensors[0].type_id, 99);
    assert_eq!(inv.tensors[0].nbytes, None);
}

/// Both paths share one parser, so a type both accept (Q5_0, and Q2_0 and
/// NVFP4, mainline ggml's ids 42 and 40) yields the same name, dims, type,
/// offset and byte count from either entry point.
#[test]
fn inventory_agrees_with_the_strict_reader_on_supported_types() {
    // dims 64x3: two 22-byte Q5_0 blocks per row, three rows = 132; dims
    // 128x3: two 18-byte Q2_0 blocks per row, three rows = 108; dims 128x3:
    // two 36-byte NVFP4 blocks per row, three rows = 216.
    for (file, dims, ty, bytes) in [
        ("q5_0.gguf", [64, 3], 6, 132),
        ("q2_0.gguf", [128, 3], 42, 108),
        ("nvfp4.gguf", [128, 3], 40, 216),
    ] {
        let p = tmp(file);
        write_gguf(&p, &[("blk.0.ffn_down.weight", &dims, ty)], bytes);

        let inv = inventory_of(&p).expect("inventory");
        let g = Gguf::open(&p).unwrap_or_else(|e| panic!("strict open of a supported type: {e}"));
        assert_eq!(inv.tensors.len(), g.tensor_count());
        let t = &inv.tensors[0];
        let s = g.tensor(0).unwrap();
        assert_eq!(t.name, s.name);
        assert_eq!(t.dims, s.dims);
        assert_eq!(t.type_id, s.ty.as_u32());
        assert_eq!(t.offset, s.offset);
        assert_eq!(t.nbytes, Some(s.nbytes));
        assert_eq!(t.nbytes, Some(bytes as u64));
    }
}

/// A tensor whose byte count passes u64 is refused by name at both entry
/// points: F32 dims [2^62 + 1, 1] need 2^64 + 4 bytes, which a wrapping
/// product counts as 4, and the file holds exactly 4.
#[test]
fn a_byte_count_past_u64_is_refused_by_name() {
    let p = tmp("too-large.gguf");
    write_gguf(&p, &[("big", &[(1 << 62) + 1, 1], 0)], 4);
    for r in both(&p) {
        match r {
            Err(LoadError::TooLarge { name, dims, ty }) => {
                assert_eq!(
                    (name.as_str(), dims, ty),
                    ("big", vec![(1 << 62) + 1, 1], GgmlType::F32)
                );
            }
            other => panic!("the tensor must be refused as too large, got {other:?}"),
        }
    }
    std::fs::remove_file(&p).unwrap();
}

/// `Gguf::data` refuses a `TensorInfo` whose offset wraps the start of its
/// slice, as it refuses one whose end runs past the mapping: offset u64::MAX
/// would start one byte before the data.
#[test]
fn data_refuses_an_offset_past_the_mapping() {
    let p = tmp("data-offset.gguf");
    write_gguf(&p, &[("f", &[4], 0)], 16);
    let g = Gguf::open(&p).expect("open");
    let mut t = g.find("f").expect("tensor f").clone();
    t.offset = u64::MAX;
    match g.data(&t) {
        Err(LoadError::OutOfBounds { name, off, .. }) => {
            assert_eq!((name.as_str(), off), ("f", u64::MAX));
        }
        other => panic!("an offset past the mapping must be refused, got {other:?}"),
    }
    std::fs::remove_file(&p).unwrap();
}

/// A header-only deepseek2 file with the given expert count and top-k and one routed
/// tensor of 2^63 bytes (F32 dims [2^61, 1]) in block 0.
fn moe_header(path: &std::path::Path, experts: u32, top_k: u32) {
    let mut arch = Vec::new();
    put_str(&mut arch, "deepseek2");
    let kvs = [
        ("general.architecture", 8, arch),
        ("deepseek2.expert_count", 4, experts.to_le_bytes().to_vec()),
        (
            "deepseek2.expert_used_count",
            4,
            top_k.to_le_bytes().to_vec(),
        ),
    ];
    write_gguf_kvs(
        path,
        &kvs,
        &[("blk.0.ffn_up_exps.weight", &[1 << 61, 1], 0)],
        0,
    );
}

/// `gguf-inventory` refuses a top-k above the expert count by its keys, with exit 1, nothing
/// on stdout and no Markdown file, before any per-token sum: 2^63 routed bytes times top-k 2
/// over 1 expert is 2^64. An expert count of 0 allows only a top-k of 0. Top-k at most the
/// expert count is reported: equal to it, all 2^63 bytes are routed; with 0 experts there is
/// no routed figure.
#[test]
fn a_top_k_above_the_expert_count_is_refused_by_name() {
    let p = tmp("top-k.gguf");
    let md = tmp("top-k.md");
    let run = |experts, top_k| {
        moe_header(&p, experts, top_k);
        std::process::Command::new(env!("CARGO_BIN_EXE_gguf-inventory"))
            .args(["--markdown".as_ref(), md.as_os_str(), p.as_os_str()])
            .output()
            .expect("run gguf-inventory")
    };
    for (experts, top_k) in [(1, 2), (0, 1)] {
        let out = run(experts, top_k);
        assert_eq!(out.status.code(), Some(1), "{out:?}");
        assert!(out.stdout.is_empty() && !md.exists(), "{out:?}");
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            format!(
                "gguf-inventory: deepseek2.expert_used_count {top_k} is greater than \
                 deepseek2.expert_count {experts}\n"
            )
        );
    }
    let out = run(2, 2);
    assert!(out.status.success(), "{out:?}");
    let routed = "routed experts/token = 9,223,372,036,854,775,808 bytes";
    assert!(String::from_utf8_lossy(&out.stdout).contains(routed));
    assert!(
        std::fs::read_to_string(&md)
            .unwrap()
            .contains("| routed experts | 9,223,372,036,854,775,808 |")
    );
    let out = run(0, 0);
    assert!(out.status.success(), "{out:?}");
    assert!(String::from_utf8_lossy(&out.stdout).contains("routed experts/token = n/a"));
    std::fs::remove_file(&p).unwrap();
    std::fs::remove_file(&md).unwrap();
}
