//! glm5next metadata gate: what the header reader `arch::glm5next` resolves
//! from the GLM-5.3-Flash file, and what the coverage check lists for it —
//! the parts no program in this tree runs yet, the work queue of the rounds
//! that build one. One contract, one test; it prints what it compared, then
//! fails with the whole list of what differs.
//!
//! `hw_`: needs the six shards on the box (`just gate-glm5next-meta`).
//! Headers only: seconds.

#[path = "common/spec_view.rs"]
mod spec_view;

#[path = "common/spec_fail_first.rs"]
mod spec_fail_first;

use std::collections::BTreeMap;
use std::fmt::Write as _;

use gguf::Split;

/// The file: unsloth's `UD-Q4_K_XL`, the quant whose routed stacks fit the
/// host tier. Its first shard holds the header and no tensor.
const GLM: &str = "/models/GLM-5.3-Flash-UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf";

// PIN(2026-09-27): the file's description as `arch::glm5next::spec` reads it, line by line (the key
// values checked against the header dump and ik's glm5next loader, src/llama-hparams.cpp).
// PIN(2026-09-27): the `hc` and latent lines re-pinned when `HcSpec` gained its kind (mHC or
// gated-residual) and `TokenPool` its rule (learned or mean pools); the values are the same.
const VIEW: &[&str] = &[
    "arch Glm5Next",
    "hidden 4096 vocab 154880 ctx_train 1048576",
    "rms_eps bits 0x3727c5ac",
    "layers 45 mtp 1",
    "hc Some(HcSpec { streams: 4, kind: Mhc { sinkhorn: 20, eps: 1e-6, mix: Own, collapse: Mean } })",
    "engram None",
    "chat pre glm4 template bytes 10648 tools None reasoning Some(ThinkSpan)",
    "[0-2] delta Kda { gate_lower_bound: -5.0 } k 64 v 64 d 128 conv 4 || dense 12288 swiglu Some(10.0) || hc",
    "[3,7,11,15,19,23,27,31,35,39,43] latent h 64 q 1536 kv 512 Absorbed { qk: 256, v: 256 } rope - qhn false out Plain win None sinks false | TokenPool { heads: 32, d: 128, top_k: 2048, pool: 4, rule: Learned { key_eps: 1e-6 } } || moe 288/8 ff 2048 swiglu Some(10.0) Sigmoid bias true norm true x2.5 hash false shared 2048 swiglu Some(10.0) gate false || hc",
    "[4-6,8-10,12-14,16-18,20-22,24-26,28-30,32-34,36-38,40-42,44] delta Kda { gate_lower_bound: -5.0 } k 64 v 64 d 128 conv 4 || moe 288/8 ff 2048 swiglu Some(10.0) Sigmoid bias true norm true x2.5 hash false shared 2048 swiglu Some(10.0) gate false || hc",
];

// PIN(2026-09-27): the next-token layer, carried and not run: ik numbers it 45, the first past the
// trunk (n_layer_kv_from_start).
// PIN(2026-09-27): re-pinned when `TokenPool` gained its rule; the values are the same.
const MTP: &[&str] = &[
    "mtp 45: latent h 64 q 1536 kv 512 Absorbed { qk: 256, v: 256 } rope - qhn false out Plain win None sinks false | TokenPool { heads: 32, d: 128, top_k: 2048, pool: 4, rule: Learned { key_eps: 1e-6 } } || moe 288/8 ff 2048 swiglu Some(10.0) Sigmoid bias true norm true x2.5 hash false shared 2048 swiglu Some(10.0) gate false || plain",
];

// PIN(2026-09-27): the file carries every key the reader reads, so it takes no default.
const DEFAULTS: &[&str] = &[];

// PIN(2026-09-27): the keys the reader does not read: listed, not refused.
const UNREAD: &[&str] = &["general.sampling.top_p", "general.sampling.temp"];

// PIN(2026-09-27): every tensor by role and type (1,412 in all, none in the first shard).
const TENSORS: &[&str] = &[
    "attn f32 315",
    "attn q8_0 405",
    "hc f32 180",
    "hc q8_0 90",
    "router f32 84",
    "ffn_norm f32 45",
    "shexp q8_0 126",
    "dense_ffn q8_0 9",
    "routed q4_K 82",
    "routed q5_K 41",
    "routed q6_K 3",
    "token_embd q8_0 1",
    "head f32 1",
    "head q8_0 1",
    "unused f32 13",
    "unused q4_K 2",
    "unused q5_K 1",
    "unused q8_0 13",
];

// PIN(2026-09-27): the coverage check's list (feature: layers). No program runs glm5next, so a
// need any program's row covers (the four hyper-connection streams, the plain shared expert) and
// a type any program's pin reads are not items; the next-token layer needs nothing.
const COVERAGE: &[&str] = &[
    "delta rule KDA: d 128, 64 heads, conv 4: 0-2,4-6,8-10,12-14,16-18,20-22,24-26,28-30,32-34,36-38,40-42,44",
    "dense SwiGLU layer: ff 12288: 0-2",
    "latent attention 512 x rope 0, absorbed qk 256 / v 256: 3,7,11,15,19,23,27,31,35,39,43",
    "token-pool indexer: 32 heads x 128, pool 4: 3,7,11,15,19,23,27,31,35,39,43",
    "a LayerNorm with a bias on the index keys: 3,7,11,15,19,23,27,31,35,39,43",
    "router: sigmoid, 288 experts, top 8, with a selection bias: 3-44",
    "q8_0 token embedding (the card reads q4_K rows)",
    "q8_0 output head (the head reads q6_K)",
    "q8_0 hyper-connection fn (the chain reads q3_K and f32): 0-44",
    "q5_K routed experts on a card: 3-11,13-43",
    "hyper-connection mix Own, collapse Mean",
    "a recurrent-state slot per sequence (delta-rule state and conv inputs)",
    "a program that runs delta-rule and attention layers in one trunk",
    "a layer program for glm5next",
    "pre-tokenizer glm4",
    "a tool-call parser for this template",
];

#[test]
#[ignore = "needs the GLM-5.3-Flash shards on the box (just gate-glm5next-meta)"]
fn hw_glm5next_spec() {
    let mut o = String::new();
    let mut b = Vec::new();
    let split = Split::open(GLM).unwrap_or_else(|e| panic!("open {GLM}: {e}"));
    let read = model::arch::spec(&split).unwrap_or_else(|e| panic!("spec of {GLM}: {e}"));
    let _ = writeln!(o, "file {GLM}");
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
    spec_view::compare(&mut o, &mut b, "next-token layers", &mtp, MTP);
    spec_view::compare(&mut o, &mut b, "defaults taken", &read.defaults, DEFAULTS);
    let unread = model::arch::glm5next::hparams::unread_keys(&split);
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
    let list = model::arch::coverage::check(&read.spec, &read.tensors);
    spec_view::compare(
        &mut o,
        &mut b,
        "coverage",
        &spec_view::items(&list),
        COVERAGE,
    );
    for (at, want) in [
        (
            "gpu-deepseek41/src/hc.rs HC_STREAMS",
            "hyper-connections of 4 streams: 0-44",
        ),
        (
            "gpu-deepseek41/src/chain/ffn.rs (the shared expert)",
            "shared expert, ff 2048: 3-44",
        ),
    ] {
        spec_fail_first::fail_first(&mut o, &mut b, &read.spec, &read.tensors, at, want);
    }
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
