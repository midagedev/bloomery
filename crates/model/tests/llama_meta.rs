//! llama metadata gate: what the header reader `arch::llama` resolves from a
//! Nanbeige4.3-33B file, what the model needs of a program, and what the
//! coverage check lists for it — exactly the two kernel gaps of a head-128
//! grouped-query layer on the pairwise rope, and neither a pre-tokenizer nor a
//! program. One contract, one test; it prints what it compared, then fails with
//! the whole list of what differs.
//!
//! `hw_`: needs the file on the box. Headers only: seconds.

#[path = "common/spec_view.rs"]
mod spec_view;

use std::collections::BTreeMap;
use std::fmt::Write as _;

use gguf::Split;
use model::arch::models::needs;
use model::placement::{Role, Unimplemented};

/// The bartowski Q4_K_M file of Nanbeige4.3-33B.
const FILE: &str = "/models/Nanbeige4.3-33B/Nanbeige_Nanbeige4.3-33B-Q4_K_M.gguf";

/// The description as `arch::llama::spec` reads the header, line by line. The
/// rope is the pairwise (NORM) one over the whole head, no layer has a q/k norm
/// or an output gate, and every layer is the same dense SwiGLU block.
const VIEW: &[&str] = &[
    "arch Llama",
    "hidden 7168 vocab 166912 ctx_train 524288",
    "rms_eps bits 0x3727c5ac",
    "layers 52 mtp 0",
    "hc None",
    "engram None",
    "chat pre llama template bytes 8417 tools Some(QwenXml) reasoning Some(ThinkSpan { open: \"<think>\", close: \"</think>\" })",
    "[0-51] gqa 56/8 x 128 rope NormTail 128 base 120000000 yarn - qk_norm false out_gate None || dense 22016 swiglu None || plain",
];

/// The keys the reader took a default for: only the pre-tokenizer, which the
/// SentencePiece tokenizer fixes whatever the file's `tokenizer.ggml.pre` says.
const DEFAULTS: &[&str] =
    &["tokenizer.ggml.pre = llama (tokenizer.ggml.model llama; the file says Some(\"default\"))"];

/// What the model needs of a program, layer by layer: the head-128 flash at
/// group 7, the pairwise rope with no head norm, the dense SwiGLU block.
const NEEDS: &[&str] = &[
    "GQA flash, head 128, group 7: 0-51",
    "rope without a QK norm: head 128, NORM tail, 128 of 128 dims: 0-51",
    "dense SwiGLU layer: ff 22016: 0-51",
];

/// The coverage check's list: empty. The group-7 flash, the NORM rope without a
/// head norm, the dense block, every weight format, the tool parser and the
/// pre-tokenizer are run, and the program is the Qwen3.5 body's.
const COVERAGE: &[&str] = &[];

/// The file's tensors by role: nine per layer on 52 layers (five attention, the
/// pre-FFN norm, three dense FFN matrices), the embedding, and the head's two
/// (the output norm and `output.weight`: the head is untied).
const ROLES: &[&str] = &[
    "attn 260",
    "ffn_norm 52",
    "dense_ffn 156",
    "token_embd 1",
    "head 2",
];

#[test]
#[ignore = "needs the Nanbeige4.3-33B file on the box"]
fn hw_llama_spec() {
    let mut o = String::new();
    let mut b = Vec::new();
    let split = Split::open(FILE).unwrap_or_else(|e| panic!("open {FILE}: {e}"));
    let read = model::arch::spec(&split).unwrap_or_else(|e| panic!("spec of {FILE}: {e}"));
    let _ = writeln!(o, "file {FILE}");
    spec_view::compare(
        &mut o,
        &mut b,
        "description",
        &spec_view::view(&read.spec),
        VIEW,
    );
    spec_view::compare(&mut o, &mut b, "defaults taken", &read.defaults, DEFAULTS);
    let wanted: Vec<Unimplemented> = needs(&read.spec)
        .into_iter()
        .map(|(need, layer)| Unimplemented {
            layer: layer.map(|l| l as usize),
            feature: need.to_string(),
        })
        .collect();
    spec_view::compare(&mut o, &mut b, "needs", &spec_view::items(&wanted), NEEDS);
    let list = model::arch::coverage::check(&read.spec, &read.tensors);
    spec_view::compare(
        &mut o,
        &mut b,
        "coverage",
        &spec_view::items(&list),
        COVERAGE,
    );
    let mut by_role: BTreeMap<Role, usize> = BTreeMap::new();
    for t in &read.tensors.tensors {
        *by_role.entry(t.role).or_default() += 1;
    }
    let roles: Vec<String> = by_role.iter().map(|(r, n)| format!("{r} {n}")).collect();
    spec_view::compare(&mut o, &mut b, "tensors by role", &roles, ROLES);
    println!("{o}");
    assert!(
        b.is_empty(),
        "{} pin(s) differ:\n  {}",
        b.len(),
        b.join("\n  ")
    );
}
