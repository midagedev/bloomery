//! qwen4exp metadata gate: what the header reader `arch::qwen35moe` resolves
//! from the Qwen3.8-Flash-Next file, and what the coverage check lists for it
//! — the parts no program in this tree runs yet, the work queue of the rounds
//! that build one. One contract, one test; it prints what it compared, then
//! fails with the whole list of what differs.
//!
//! `hw_`: needs the four shards on the box (`just gate-qwen4exp-meta`).
//! Headers only: seconds.

#[path = "common/spec_view.rs"]
mod spec_view;

use std::collections::BTreeMap;
use std::fmt::Write as _;

use gguf::Split;

/// The file: unsloth's `UD-Q4_K_XL`. Its first shard holds the header and no
/// tensor.
const Q38: &str = "/models/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";

// PIN(2026-09-27): the file's description as `arch::qwen35moe::spec` reads it, line by line (the key
// values checked against the header dump and llama.cpp's loader, src/models/qwen4exp.cpp:26-147).
const VIEW: &[&str] = &[
    "arch Qwen4Exp",
    "hidden 2560 vocab 248320 ctx_train 262144",
    "rms_eps bits 0x358637bd",
    "layers 48 mtp 0",
    "hc Some(HcSpec { streams: 4, kind: Gated { rank: 320 } })",
    "engram Some(EngramSpec { heads: 8, max_ngram: 3, key_length: 160, rule: Ple { eos: 248044, image: Some(248056), conv: 4, dilation: 3 } })",
    "chat pre qwen35 template bytes 9993 tools None reasoning None",
    "[0,2,4-6,8-10,12-14,16-18,20-22,24-26,28-30,32-34,36-38,40-42,44-46] delta Gdn { khead_map: Tiled, gate: Sigmoid } k 16 v 48 d 128 conv 4 || moe 512/10 ff 640 swiglu None Softmax bias false norm true x1 hash false shared 640 swiglu None gate true || hc",
    "[1] delta Gdn { khead_map: Tiled, gate: Sigmoid } k 16 v 48 d 128 conv 4 || moe 512/10 ff 640 swiglu None Softmax bias false norm true x1 hash false shared 640 swiglu None gate true || hc [Ple]",
    "[3,7,11,15,19,23,27,31,35,39,43,47] gqa 24/2 x 256 rope Imrope { sections: [11, 11, 10, 0] } 64 base 10000000 yarn - qk_norm true out_gate true | TokenPool { heads: 4, d: 128, top_k: 2048, pool: 4, rule: Mean { rope: Rope { mode: Imrope { sections: [11, 11, 10, 0] }, dims: 64, base: 10000000.0, yarn: None } } } || moe 512/10 ff 640 swiglu None Softmax bias false norm true x1 hash false shared 640 swiglu None gate true || hc",
];

// PIN(2026-09-27): the keys the reader took a default for: the router's constants, which the file
// does not carry.
const DEFAULTS: &[&str] = &[
    "expert_gating_func = softmax (qwen4exp.cpp:992-1001)",
    "expert_weights_norm = true (qwen4exp.cpp:992-1001)",
    "expert_weights_scale = 1 (qwen4exp.cpp:992-1001)",
];

// PIN(2026-09-27): every tensor by role and type (1,224 in all, none in the first shard); the PLE
// table is the site's, the output hyper-connection head the head's.
const TENSORS: &[&str] = &[
    "attn bf16 24",
    "attn f32 264",
    "attn q8_0 156",
    "hc f32 192",
    "hc q8_0 192",
    "router f32 48",
    "shexp f32 48",
    "shexp q8_0 144",
    "engram_dense q8_0 2",
    "engram_gain f32 4",
    "engram_table iq4_nl 1",
    "routed q4_K 94",
    "routed q5_1 43",
    "routed q5_K 2",
    "routed q8_0 5",
    "token_embd q8_0 1",
    "head f32 1",
    "head q8_0 3",
];

// PIN(2026-09-27): the coverage check's list (feature: layers). No program runs qwen4exp, so a
// type any program's pin reads (the q8_0 attention matrices) is not an item. The routed q5_1 and
// q8_0 stacks are not items either: the check counts a card format (`CardFormat::of_routed`), not
// a card expert kernel for it, and no card `_sel` reads either type yet. No pin reads the PLE
// table's type. The gated-residual hyper-connections, the mean-pool indexer and the PLE site are
// needs of their own, which no row for mHC, the token pool or the engram covers.
// PIN(2026-09-27): the router and the GQA flash rows left the list when the shape registry gained
// their serving rows (shapes p1-4: the `_512` router and the `_256_p4` flash entries serve them).
// PIN(2026-09-27): five items left the list with GLM-5.3-Flash's first program, whose rows and type
// pins the whole-tree listing now counts: the q8_0 output head, token embedding and hyper-connection
// fn (its pins read q8_0), the recurrent-state slot (`gpu-glm5next/src/kda.rs`) and the program that
// runs delta-rule and attention layers in one trunk (`gpu-glm5next/src/program.rs`).
const COVERAGE: &[&str] = &[
    "delta rule GDN: d 128, 16 k-heads and 48 v-heads (tiled), conv 4, sigmoid output gate: 0-2,4-6,8-10,12-14,16-18,20-22,24-26,28-30,32-34,36-38,40-42,44-46",
    "shared expert, ff 640, with a sigmoid gate: 0-47",
    "gated-residual hyper-connections of 4 streams, rank 320: 0-47",
    "a PLE site: a gate of 2560-value rows, a conv of 4 taps 3 apart: 1",
    "per-head QK norm plus rope: head 256, IMROPE [11, 11, 10, 0], 64 of 256 dims: 3,7,11,15,19,23,27,31,35,39,43,47",
    "attention output gate: sigmoid, interleaved with q: 3,7,11,15,19,23,27,31,35,39,43,47",
    "mean-pool indexer: 4 heads x 128, pool 4, RMS-normed keys roped (64 dims) at the pool start, heads summed: 3,7,11,15,19,23,27,31,35,39,43,47",
    "q5_K routed experts on a card: 2",
    "bf16 attention matrices (the body reads q4_K and q6_K): 3,7,11,15,19,23,27,31,35,39,43,47",
    "the gated-residual hyper-connection head output_hc_*, rank 320",
    "a text prompt carrying the image token 248056 refused by name",
    "a layer program for qwen4exp",
    "pre-tokenizer qwen35",
    "a tool-call parser for this template",
];

#[test]
#[ignore = "needs the Qwen3.8-Flash-Next shards on the box (just gate-qwen4exp-meta)"]
fn hw_qwen4exp_spec() {
    let mut o = String::new();
    let mut b = Vec::new();
    let split = Split::open(Q38).unwrap_or_else(|e| panic!("open {Q38}: {e}"));
    let read = model::arch::spec(&split).unwrap_or_else(|e| panic!("spec of {Q38}: {e}"));
    let _ = writeln!(o, "file {Q38}");
    spec_view::compare(
        &mut o,
        &mut b,
        "description",
        &spec_view::view(&read.spec),
        VIEW,
    );
    spec_view::compare(&mut o, &mut b, "defaults taken", &read.defaults, DEFAULTS);
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
