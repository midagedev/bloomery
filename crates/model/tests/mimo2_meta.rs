//! mimo2 metadata gate: what the header reader `arch::mimo2` resolves from
//! the MiMo-V2.6-Flash-RL file, and what the coverage check lists for it —
//! the parts no program in this tree runs yet, the work queue of the rounds
//! that build one. One contract, one test; it prints what it compared, then
//! fails with the whole list of what differs.
//!
//! `hw_`: needs the RL shards on the box (`just gate-mimo2-meta`). Headers
//! only: seconds.

#[path = "common/spec_view.rs"]
mod spec_view;

use std::collections::BTreeMap;
use std::fmt::Write as _;

use gguf::{Gguf, Split, Value};

/// The file: ggml's RL conversion, MXFP4 experts and a Q8_0 dense tier. Its
/// first shard holds the header and no tensor.
const MIMO: &str = "/models/MiMo-V2.6-Flash-RL/MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf";

/// The separate MTP file, whose 51 blocks carry only the next-token three:
/// a trunk's tensors it does not carry, refused by name until the draft
/// round reads it beside a target.
const MTP: &str = "/models/MiMo-V2.6-Flash-RL/mtp-MiMo-V2.6-Flash-RL-MXFP4.gguf";

// PIN(2026-10-07): the file's description as `arch::mimo2::spec` reads it, line by line (the key
// values checked against the shard-0 header dump and llama.cpp's mimo2 loader, src/models/mimo2.
// cpp). The value head is narrower than the key head (192/128), the KV-head count and the rope
// base are the layer's own (4 and 1e7 on the nine full layers, 8 and 1e4 on the window ones).
const VIEW: &[&str] = &[
    "arch MiMo2",
    "hidden 4096 vocab 152576 ctx_train 1048576",
    "rms_eps bits 0x358637bd",
    "layers 48 mtp 0",
    "hc None",
    "engram None",
    "chat pre qwen2 template bytes 3867 tools Some(QwenXml) reasoning Some(ThinkSpan)",
    "[0] gqa 64/4 x 192 v 128 rope Neox 64 base 10000000 yarn - qk_norm false out_gate false vscale 0.707 || dense 16384 swiglu None || plain",
    "[5,11,17,23,29,35,41,47] gqa 64/4 x 192 v 128 rope Neox 64 base 10000000 yarn - qk_norm false out_gate false vscale 0.707 || moe 256/8 ff 2048 swiglu None Sigmoid bias true norm true x1 hash false || plain",
    "[1-4,6-10,12-16,18-22,24-28,30-34,36-40,42-46] gqa 64/8 x 192 v 128 rope Neox 64 base 10000 yarn - qk_norm false out_gate false win 128 sinks vscale 0.707 || moe 256/8 ff 2048 swiglu None Sigmoid bias true norm true x1 hash false || plain",
];

// PIN(2026-10-07): the RL file carries no next-token layer; the separate `mtp-` file does.
const MTP_LAYERS: &[&str] = &[];

// PIN(2026-10-07): two keys the loader reads with a fallback (absent nextn count; the scale node
// the graph skips at 1).
const DEFAULTS: &[&str] = &[
    "nextn_predict_layers = 0 (absent)",
    "expert_weights_scale = 1 (absent, no scale node)",
];

// PIN(2026-10-07): the keys the reader does not read: listed, not refused.
const UNREAD: &[&str] = &["general.sampling.temp", "general.sampling.top_p"];

// PIN(2026-10-07): every tensor by role and type (472 in all, none in the first shard).
const TENSORS: &[&str] = &[
    "attn f32 87",
    "attn q8_0 96",
    "dense_ffn q8_0 3",
    "ffn_norm f32 48",
    "head f32 1",
    "head q8_0 1",
    "routed mxfp4 141",
    "router f32 94",
    "token_embd q8_0 1",
];

// PIN(2026-10-07): the KV cache the description asks a layer to hold, in values a position: the
// full layers' 4 heads of 320 (key 192 + value 128; 2 B/value in the f16 cache) and the window
// layers' 8 — the window layers' ring holds `attention.sliding_window` rows alone.
const KV: &[&str] = &[
    "kv 4x320 values: 0,5,11,17,23,29,35,41,47",
    "kv 8x320 values: 1-4,6-10,12-16,18-22,24-28,30-34,36-40,42-46",
];

// PIN(2026-10-07): the coverage check's list (feature: layers). No program runs a mimo2 file yet,
// so a need is an item unless a row of any program covers it: the dense block (Body35's dense
// path, 16384 a multiple of its 256) and the Q8_0/F32 tensors other programs' pins read are not
// items; the flash at head 192 with the value head split from it, the rope without a QK norm, the
// sigmoid router at a width no body is built for (the 288 of GLM's) and the mxfp4 stacks are. The
// flash needs name the value head, the window, the sinks and the value scale, which no flash row runs.
const COVERAGE: &[&str] = &[
    "GQA flash, head 192, value 128, group 16, value scale: 0,5,11,17,23,29,35,41,47",
    "GQA flash, head 192, value 128, group 8, window 128, sinks, value scale: 1-4,6-10,12-16,18-22,24-28,30-34,36-40,42-46",
    "rope without a QK norm: head 192, NeoX, 64 of 192 dims: 0-47",
    "router: sigmoid, 256 experts, top 8, with a selection bias: 1-47",
    "mxfp4 routed experts on a card: 1-47",
    "mxfp4 routed experts gate and up (the body reads q4_K): 1-47",
    "mxfp4 routed experts down (the body reads q4_K and q6_K): 1-47",
    "a layer program for mimo2",
];

#[test]
#[ignore = "needs the MiMo-V2.6-Flash-RL shards on the box (just gate-mimo2-meta)"]
fn hw_mimo2_spec() {
    let mut o = String::new();
    let mut b = Vec::new();
    let split = Split::open(MIMO).unwrap_or_else(|e| panic!("open {MIMO}: {e}"));
    let read = model::arch::spec(&split).unwrap_or_else(|e| panic!("spec of {MIMO}: {e}"));
    let _ = writeln!(o, "file {MIMO}");
    spec_view::compare(
        &mut o,
        &mut b,
        "description",
        &spec_view::view(&read.spec),
        VIEW,
    );
    let n_trunk = read.spec.layers.len();
    let mtp: Vec<String> = (n_trunk..)
        .zip(&read.spec.mtp)
        .map(|(l, layer)| format!("mtp {l}: {}", spec_view::layer_line(layer)))
        .collect();
    spec_view::compare(&mut o, &mut b, "next-token layers", &mtp, MTP_LAYERS);
    spec_view::compare(&mut o, &mut b, "defaults taken", &read.defaults, DEFAULTS);
    let unread = model::arch::mimo2::hparams::unread_keys(&split);
    spec_view::compare(&mut o, &mut b, "keys not read", &unread, UNREAD);
    let mut by_role: BTreeMap<(String, String), usize> = BTreeMap::new();
    for t in &read.tensors.tensors {
        *by_role
            .entry((t.role.to_string(), t.ty.to_string()))
            .or_default() += 1;
    }
    let tensors: Vec<String> = by_role
        .iter()
        .map(|((role, ty), n)| format!("{role} {ty} {n}"))
        .collect();
    let _ = writeln!(o, "tensors {}", read.tensors.tensors.len());
    spec_view::compare(
        &mut o,
        &mut b,
        "tensors by role and type",
        &tensors,
        TENSORS,
    );
    let kv = kv_values(&read.spec);
    spec_view::compare(&mut o, &mut b, "kv values a layer holds", &kv, KV);
    let list = model::arch::coverage::check(&read.spec, &read.tensors);
    spec_view::compare(
        &mut o,
        &mut b,
        "coverage",
        &spec_view::items(&list),
        COVERAGE,
    );
    let err = model::placement::PlacementError::Unimplemented(list);
    let _ = writeln!(o, "as the engine refuses it: {err}");
    println!("{o}");
    assert!(
        b.is_empty(),
        "{} pin(s) differ:\n  {}",
        b.len(),
        b.join("\n  ")
    );
}

/// Each run of layers that hold one KV width, in values a position: the
/// KV-head count times the key and value head widths.
fn kv_values(spec: &model::arch::models::ModelSpec) -> Vec<String> {
    use model::arch::models::Mixer;
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    for (l, layer) in spec.layers.iter().enumerate() {
        let Mixer::Gqa(g) = &layer.mixer else {
            panic!("a mimo2 layer that is not GQA");
        };
        let line = format!("kv {}x{} values", g.kv_heads, g.head_dim + g.value_dim);
        match groups.iter_mut().find(|(k, _)| *k == line) {
            Some((_, ls)) => ls.push(l),
            None => groups.push((line, vec![l])),
        }
    }
    groups
        .into_iter()
        .map(|(line, ls)| format!("{line}: {}", spec_view::ranges_of(&ls)))
        .collect()
}

/// A doctored header — the RL shard-0's own keys with the KV-head array one
/// short — is refused by the array's name: the per-layer arrays are the
/// file's fact, and a wrong length is never read as a broadcast.
#[test]
#[ignore = "needs the MiMo-V2.6-Flash-RL shards on the box (just gate-mimo2-meta)"]
fn hw_mimo2_short_kv_array_refused() {
    let shard0 = Gguf::open(MIMO).expect("the metadata shard opens");
    let kv = doctored_kv(&shard0, "mimo2.attention.head_count_kv", 47);
    let path = std::env::temp_dir().join(format!(
        "bloomery-mimo2-short-kv-{}.gguf",
        std::process::id()
    ));
    std::fs::write(&path, kv).expect("write the doctored header");
    let split = Split::open(&path).expect("the doctored header opens");
    let err = model::arch::spec(&split)
        .expect_err("a short per-layer array is refused")
        .to_string();
    let _ = std::fs::remove_file(&path);
    assert!(
        err.contains("mimo2.attention.head_count_kv")
            && err.contains("has 47 values for 48 layers"),
        "{err}"
    );
}

/// The separate MTP file, whose blocks carry only the next-token layers, is
/// refused by the first trunk tensor it lacks — the draft round's reader
/// opens it beside a target.
#[test]
#[ignore = "needs the MiMo-V2.6-Flash-RL mtp file on the box (just gate-mimo2-meta)"]
fn hw_mimo2_mtp_file_is_refused_until_its_round() {
    let split = Split::open(MTP).unwrap_or_else(|e| panic!("open {MTP}: {e}"));
    let err = model::arch::spec(&split)
        .expect_err("a next-token-only file has no trunk")
        .to_string();
    assert!(
        err.contains("tensor blk.0.attn_norm.weight: is not in the file"),
        "{err}"
    );
}

/// `shard`'s metadata rewritten: the `split.*` keys dropped, so the copy is
/// a one-shard file of no tensor, and the array `key` left with its first
/// `len` values.
fn doctored_kv(shard: &Gguf, key: &str, len: usize) -> Vec<u8> {
    let kv: Vec<(String, Value)> = shard
        .iter_kv()
        .filter(|(k, _)| !k.starts_with("split."))
        .map(|(k, v)| {
            if k != key {
                return (k.to_string(), v.clone());
            }
            let Value::Array(items) = v else {
                panic!("{key} is not an array");
            };
            assert!(items.len() > len, "{key} holds {} values", items.len());
            (
                k.to_string(),
                Value::Array(items.iter().take(len).cloned().collect()),
            )
        })
        .collect();
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    b.extend_from_slice(&(kv.len() as u64).to_le_bytes());
    for (k, v) in &kv {
        put_str(&mut b, k);
        put_value(&mut b, v);
    }
    b
}

/// One metadata value as the file's bytes: the type tag then the value, an
/// array's element tag before its count.
fn put_value(b: &mut Vec<u8>, v: &Value) {
    b.extend_from_slice(&tag_of(v).to_le_bytes());
    put_raw(b, v);
}

/// The value's bytes without its type tag.
fn put_raw(b: &mut Vec<u8>, v: &Value) {
    match v {
        Value::U8(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::I8(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::U16(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::I16(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::U32(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::I32(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::F32(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::Bool(x) => b.push(u8::from(*x)),
        Value::String(s) => put_str(b, s),
        Value::Array(items) => {
            let Some(first) = items.first() else {
                panic!("an empty array has no element tag to write");
            };
            b.extend_from_slice(&tag_of(first).to_le_bytes());
            b.extend_from_slice(&(items.len() as u64).to_le_bytes());
            items.iter().for_each(|x| put_raw(b, x));
        }
        Value::U64(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::I64(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::F64(x) => b.extend_from_slice(&x.to_le_bytes()),
    }
}

/// The type tag a value is written with.
fn tag_of(v: &Value) -> u32 {
    match v {
        Value::U8(_) => 0,
        Value::I8(_) => 1,
        Value::U16(_) => 2,
        Value::I16(_) => 3,
        Value::U32(_) => 4,
        Value::I32(_) => 5,
        Value::F32(_) => 6,
        Value::Bool(_) => 7,
        Value::String(_) => 8,
        Value::Array(_) => 9,
        Value::U64(_) => 10,
        Value::I64(_) => 11,
        Value::F64(_) => 12,
    }
}

/// One length-prefixed string.
fn put_str(b: &mut Vec<u8>, s: &str) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
}
