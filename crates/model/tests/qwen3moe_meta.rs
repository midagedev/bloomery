//! Qwen3-MoE metadata gate: what `arch::qwen3moe` resolves from the file's
//! headers, against the file and against the graph ik built from it. Two
//! contracts, one test each; each prints what it compared, then fails with the
//! whole list of what differs.
//!
//! 1. `hw_qwen3moe_hparams_match_ik` — every `Hparams` field equals ik's. A
//!    value ik's graph shows is pinned from the oracle manifest's node shapes
//!    and ops (`ref_qwen3moe`, a 5-token CPU dump; ik lines are
//!    `src/graphs/build_qwen3.cpp` and `src/llama-build-context.cpp`'s
//!    `llm_build_std_moe_ffn`); a value that only reaches an op's parameters
//!    (θ, ε, the rope's width and mode) is pinned from the header with the ik
//!    line that reads it.
//! 2. `hw_qwen3moe_tensor_inventory` — every tensor of the file has a role, the
//!    file holds exactly the names `names` gives on every layer, and each name
//!    is of the types this chain's kernels load (gate/up Q4_K; down Q4_K or
//!    Q6_K; attn_q/k/output Q4_K; attn_v Q4_K or Q6_K; output Q6_K; token_embd
//!    Q4_K; norms and router F32).
//!
//! `hw_`: needs the file and the oracle set on the box (`just
//! gate-qwen3moe-meta`, which picks the qwen3moe tool profile, so
//! `BLOOMERY_REF_MODEL` is this file). Headers and a manifest only: seconds.

use std::collections::{BTreeMap, HashMap};
use std::fmt::{Debug, Write as _};
use std::path::PathBuf;

use gguf::{GgmlType, Split};
use model::arch::qwen3moe::hparams::{Hparams, RopeMode, Score};
use model::arch::qwen3moe::{names, roles};
use model::placement::Role;

/// The oracle set of the 5-token prefill.
const SET: &str = "ref_qwen3moe";

/// The file the tool profile resolved, refused unless it declares qwen3moe.
fn open() -> (Split, Hparams) {
    let path = std::env::var("BLOOMERY_REF_MODEL").unwrap_or_else(|_| {
        panic!("BLOOMERY_REF_MODEL is unset: run through `just gate-qwen3moe-meta`")
    });
    let split = Split::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    assert_eq!(
        split.architecture(),
        Some("qwen3moe"),
        "{path} is not a qwen3moe file; the recipe picks BLOOMERY_MODEL=qwen3moe"
    );
    let hp = Hparams::read(&split).unwrap_or_else(|e| panic!("hyperparameters of {path}: {e}"));
    (split, hp)
}

/// One node row of a manifest: its type, `ne`, op and first source.
struct Node {
    ne: [u64; 4],
    op: String,
    src0: String,
}

/// The `tensor` rows of `set`'s manifest, occurrence 0, by name; the
/// manifest must carry its completion trailer and name qwen3moe.
fn nodes(set: &str) -> HashMap<String, Node> {
    let base = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into());
    let path = PathBuf::from(base).join(set).join("MANIFEST.tsv");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "no oracle at {} ({e}); the lead dumps it with `just dump-ref-qwen3moe`",
            path.display()
        )
    });
    assert!(
        text.lines().any(|l| l.starts_with("# complete\t")),
        "{} has no completion trailer",
        path.display()
    );
    assert!(
        text.lines().any(|l| l == "# arch\tqwen3moe"),
        "{} is not a qwen3moe set",
        path.display()
    );
    let mut out = HashMap::new();
    for line in text.lines().filter(|l| l.starts_with("tensor\t")) {
        let f: Vec<&str> = line.split('\t').collect();
        if f.get(2) != Some(&"0") || f.len() < 15 {
            continue;
        }
        let ne = |i: usize| f[i].parse().unwrap_or_else(|e| panic!("{line}: {e}"));
        out.insert(
            f[1].to_string(),
            Node {
                ne: [ne(4), ne(5), ne(6), ne(7)],
                op: f[10].to_string(),
                src0: f[13].to_string(),
            },
        );
    }
    out
}

/// One compared value: a line of the printed table, and an entry of `bad`
/// when it differs.
fn row<T: PartialEq + Debug>(
    out: &mut String,
    bad: &mut Vec<String>,
    what: &str,
    got: T,
    want: T,
    source: &str,
) {
    let mark = if got == want { "ok " } else { "BAD" };
    let _ = writeln!(out, "  {mark} {what:<28} {got:?}  [{source}]");
    if got != want {
        bad.push(format!("{what}: ours {got:?}, ik {want:?} [{source}]"));
    }
}

fn fail_if_bad(out: &str, bad: &[String]) {
    println!("{out}");
    assert!(
        bad.is_empty(),
        "{} value(s) differ:\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}

/// `name`'s node, or a panic naming it.
fn node<'a>(g: &'a HashMap<String, Node>, name: &str) -> &'a Node {
    g.get(name)
        .unwrap_or_else(|| panic!("the oracle graph has no node {name}"))
}

#[test]
#[ignore = "needs the qwen3moe file and ref_qwen3moe on the box"]
fn hw_qwen3moe_hparams_match_ik() {
    let (split, hp) = open();
    let g = nodes(SET);
    let n_tok = node(&g, "inp_embd").ne[1];
    let (mut out, mut bad) = (String::new(), Vec::new());
    let o = &mut out;
    let b = &mut bad;
    let _ = writeln!(o, "qwen3moe hparams vs ik ({SET}, {n_tok} tokens):");

    // From the graph: the layer count is the l_out chain's length.
    let layers = (0..)
        .take_while(|l| g.contains_key(&format!("l_out-{l}")))
        .count();
    row(o, b, "n_layer", hp.n_layer, layers, "l_out-N nodes");
    row(
        o,
        b,
        "n_embd",
        hp.n_embd as u64,
        node(&g, "attn_norm-0").ne[0],
        "attn_norm-0 ne0",
    );
    let q = node(&g, "Qcur_normed-0");
    let k = node(&g, "Kcur_normed-0");
    row(
        o,
        b,
        "head_dim",
        hp.head_dim as u64,
        q.ne[0],
        "Qcur_normed-0 ne0",
    );
    row(
        o,
        b,
        "head_dim (key)",
        hp.head_dim as u64,
        k.ne[0],
        "Kcur_normed-0 ne0",
    );
    row(
        o,
        b,
        "n_head",
        hp.n_head as u64,
        q.ne[1],
        "Qcur_normed-0 ne1",
    );
    row(
        o,
        b,
        "n_head_kv",
        hp.n_head_kv as u64,
        k.ne[1],
        "Kcur_normed-0 ne1",
    );
    row(
        o,
        b,
        "QK norm op",
        q.op.as_str(),
        "FUSED_RMS_NORM",
        "Qcur_normed-0 op",
    );
    let rope = node(&g, "Qcur_roped-0");
    row(
        o,
        b,
        "rope after norm",
        rope.src0.as_str(),
        "Qcur_normed-0",
        "Qcur_roped-0 src0",
    );
    row(
        o,
        b,
        "n_vocab",
        hp.n_vocab as u64,
        node(&g, "result_output").ne[0],
        "result_output ne0",
    );
    let probs = node(&g, "ffn_moe_probs-0");
    row(
        o,
        b,
        "n_expert",
        hp.experts.n_expert as u64,
        probs.ne[0],
        "ffn_moe_probs-0 ne0",
    );
    row(
        o,
        b,
        "score",
        hp.experts.score == Score::Softmax,
        probs.op == "SOFT_MAX",
        "ffn_moe_probs-0 op",
    );
    row(
        o,
        b,
        "n_used",
        hp.experts.n_used as u64,
        node(&g, "ffn_moe_topk-0").ne[0],
        "ffn_moe_topk-0 ne0",
    );
    row(
        o,
        b,
        "weights_norm",
        hp.experts.weights_norm,
        node(&g, "ffn_moe_weights_norm-0").op == "DIV",
        "ffn_moe_weights_norm-0 op",
    );
    row(
        o,
        b,
        "ff",
        hp.experts.ff as u64,
        node(&g, "ffn_moe_gate_par-0").ne[0],
        "ffn_moe_gate_par-0 ne0",
    );
    let every_layer_routes = (0..layers).all(|l| g.contains_key(&format!("ffn_moe_probs-{l}")));
    row(
        o,
        b,
        "no dense layer",
        true,
        every_layer_routes,
        "ffn_moe_probs-N on every layer",
    );

    // From the header, with the ik line that reads it: values that reach an
    // op's parameters only.
    let f32_key = |k: &str| split.arch_get_f32(k).unwrap_or(f32::NAN);
    let u64_key = |k: &str| split.arch_get_u64(k).unwrap_or(u64::MAX);
    row(
        o,
        b,
        "rope base",
        hp.rope.base,
        f32_key("rope.freq_base"),
        "llama-hparams.cpp:203",
    );
    let key_length = u64_key("attention.key_length");
    let rope_dims = split
        .arch_get_u64("rope.dimension_count")
        .unwrap_or(key_length);
    row(
        o,
        b,
        "rope dims",
        hp.rope.dims as u64,
        rope_dims,
        "llama-hparams.cpp:236-238",
    );
    row(
        o,
        b,
        "rope mode",
        hp.rope.mode,
        RopeMode::Neox,
        "llama.cpp:9737, 9765",
    );
    row(
        o,
        b,
        "rms_eps",
        hp.rms_eps,
        f32_key("attention.layer_norm_rms_epsilon"),
        "llama-hparams.cpp:559",
    );
    row(
        o,
        b,
        "n_ctx_train",
        hp.n_ctx_train as u64,
        u64_key("context_length"),
        "header",
    );
    fail_if_bad(&out, &bad);
}

/// The types each name may carry in this chain.
fn allowed(stem: &str) -> &'static [GgmlType] {
    match stem {
        "ffn_down_exps.weight" | "attn_v.weight" => &[GgmlType::Q4_K, GgmlType::Q6_K],
        "ffn_gate_exps.weight"
        | "ffn_up_exps.weight"
        | "attn_q.weight"
        | "attn_k.weight"
        | "attn_output.weight"
        | "token_embd.weight" => &[GgmlType::Q4_K],
        "output.weight" => &[GgmlType::Q6_K],
        _ => &[GgmlType::F32],
    }
}

#[test]
#[ignore = "needs the qwen3moe file on the box"]
fn hw_qwen3moe_tensor_inventory() {
    let (split, hp) = open();
    let mut out = String::new();
    let mut bad = Vec::new();
    let model = roles::classify(&split, &hp).unwrap_or_else(|e| panic!("classify: {e}"));
    let mut by_role: BTreeMap<Role, usize> = BTreeMap::new();
    for t in &model.tensors {
        *by_role.entry(t.role).or_default() += 1;
    }
    let _ = writeln!(
        out,
        "qwen3moe tensors: {} classified, by role {by_role:?}",
        model.tensors.len()
    );

    let mut want: Vec<String> = vec![names::token_embd(), names::output_norm(), names::output()];
    for l in 0..hp.n_layer {
        want.extend(names::LAYER.iter().map(|s| format!("blk.{l}.{s}")));
    }
    let in_file: Vec<&str> = split.iter_tensors().map(|(_, t)| t.name.as_str()).collect();
    for name in &want {
        if split.find(name).is_none() {
            bad.push(format!("{name}: named by the table, not in the file"));
        }
    }
    let extra: Vec<&&str> = in_file
        .iter()
        .filter(|n| !want.iter().any(|w| w == **n))
        .collect();
    if !extra.is_empty() {
        bad.push(format!("in the file, not named by the table: {extra:?}"));
    }
    let _ = writeln!(
        out,
        "  names: {} named, {} in the file",
        want.len(),
        in_file.len()
    );

    // Types per stem, counted.
    let mut types: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    for (_, t) in split.iter_tensors() {
        let stem = t
            .name
            .strip_prefix("blk.")
            .and_then(|r| r.split_once('.'))
            .map_or(t.name.as_str(), |(_, s)| s);
        *types
            .entry(stem.to_string())
            .or_default()
            .entry(t.ty.to_string())
            .or_default() += 1;
        if !allowed(stem).contains(&t.ty) {
            bad.push(format!(
                "{}: type {}, allowed {:?}",
                t.name,
                t.ty,
                allowed(stem)
            ));
        }
    }
    for (stem, tys) in &types {
        let _ = writeln!(out, "  {stem:<24} {tys:?}");
    }
    fail_if_bad(&out, &bad);
}
