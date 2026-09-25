//! DeepSeek-V4-Flash inventory gate: what `arch::deepseek41` reads from the
//! V4-Flash file (`general.architecture` `deepseek4`), from its headers alone —
//! no tensor bytes, no GPU, no lease, seconds. Three contracts, one test each;
//! each prints what it compared, then fails with the whole list of what
//! differs from `common/deepseek4_pins.rs`.
//!
//! 1. `hw_deepseek4_hparams_and_kinds` — the model values and every layer's
//!    kind: which layers select through their own index keys (ratio 4), which
//!    attend their stream whole (ratio 128), which route by the token table,
//!    and what each compressor is (gated, position table, overlapping groups).
//! 2. `hw_deepseek4_tensor_inventory` — the tensor count and bytes, bytes per
//!    role family and per type, one expert's bytes on every routed layer.
//! 3. `hw_deepseek4_plans` — the placement of design §5 (a), (b) and the gate
//!    placement on this file: card dense, rounding and KV, the planner's card
//!    experts (none while no card format loads IQ3_XXS or MXFP4), the experts
//!    the card budget would hold at the file's bytes [derived], the host line;
//!    and the features the engine refuses the file for, every one.
//!
//! `hw_`: needs the file on the box (`just gate-deepseek4-meta`, which picks the
//! deepseek4 tool profile, so `BLOOMERY_REF_MODEL` is this file).

#[path = "common/deepseek4_pins.rs"]
mod pins;

use std::collections::BTreeMap;
use std::fmt::{Debug, Write as _};

use gguf::{GgmlType, Split};
use model::arch::deepseek41::hparams::{Collapse, Hparams, Model, Score};
use model::arch::deepseek41::place::PlanInputs;
use model::placement::{self, KvBytes, Plan, Role, workstation};
use pins::{CardPin, LayerSet, PlanPin};

/// The file the tool profile resolved, refused unless it declares deepseek4.
fn open() -> (Split, PlanInputs) {
    let path = std::env::var("BLOOMERY_REF_MODEL").unwrap_or_else(|_| {
        panic!("BLOOMERY_REF_MODEL is unset: run through `just gate-deepseek4-meta`")
    });
    let split = Split::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    assert_eq!(
        split.architecture(),
        Some("deepseek4"),
        "{path} is not a deepseek4 file; the recipe picks BLOOMERY_MODEL=deepseek4"
    );
    let inputs =
        PlanInputs::describe(&split).unwrap_or_else(|e| panic!("plan inputs of {path}: {e}"));
    (split, inputs)
}

/// One compared value: a line of the printed table, and an entry of `bad`
/// when it differs.
fn row<T: PartialEq + Debug>(out: &mut String, bad: &mut Vec<String>, what: &str, got: T, want: T) {
    let mark = if got == want { "ok " } else { "BAD" };
    let _ = writeln!(out, "  {mark} {what:<34} {got:?}");
    if got != want {
        bad.push(format!("{what}: got {got:?}, want {want:?}"));
    }
}

/// Print `out`, then fail with every entry of `bad`.
fn fail_if_bad(out: &str, bad: &[String]) {
    println!("{out}");
    assert!(
        bad.is_empty(),
        "{} pin(s) differ:\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}

/// The layers of `hp` for which `pick` holds.
fn layers(
    hp: &Hparams,
    pick: impl Fn(&model::arch::deepseek41::hparams::LayerKind) -> bool,
) -> Vec<usize> {
    hp.layers
        .iter()
        .enumerate()
        .filter(|(_, k)| pick(k))
        .map(|(l, _)| l)
        .collect()
}

fn set_row(out: &mut String, bad: &mut Vec<String>, set: &LayerSet, got: Vec<usize>) {
    row(out, bad, set.what, got, set.layers.to_vec());
}

#[test]
#[ignore = "needs the V4-Flash file (just gate-deepseek4-meta)"]
fn hw_deepseek4_hparams_and_kinds() {
    let (_, inputs) = open();
    let hp = &inputs.hp;
    let (mut o, mut b) = (String::new(), Vec::new());
    let (o, b) = (&mut o, &mut b);
    let _ = writeln!(o, "model values");
    row(o, b, "model", hp.model, Model::Deepseek4);
    row(o, b, "n_layer", hp.n_layer, 43);
    row(o, b, "n_embd", hp.n_embd, 4096);
    row(
        o,
        b,
        "n_head / n_head_kv",
        (hp.n_head, hp.n_head_kv),
        (64, 1),
    );
    row(
        o,
        b,
        "head_dim / rope_dims",
        (hp.head_dim, hp.rope_dims),
        (512, 64),
    );
    row(o, b, "q_lora_rank", hp.q_lora_rank, 1024);
    row(
        o,
        b,
        "o_groups / o_lora_rank",
        (hp.o_groups, hp.o_lora_rank),
        (8, 1024),
    );
    row(o, b, "window", hp.window, 128);
    row(o, b, "rms_eps", hp.rms_eps, 1e-6);
    row(o, b, "q_head_norm", hp.q_head_norm, true);
    row(o, b, "n_vocab", hp.n_vocab, 129_280);
    row(o, b, "n_ctx_train", hp.n_ctx_train, 1_048_576);
    row(
        o,
        b,
        "csa_ratio / hca_ratio",
        (hp.csa_ratio, hp.hca_ratio),
        (4, 128),
    );
    let ix = hp.indexer;
    row(
        o,
        b,
        "indexer n_head / head_dim / top_k",
        (ix.n_head, ix.head_dim, ix.top_k),
        (64, 128, 512),
    );
    let hc = hp.hc;
    row(
        o,
        b,
        "hc streams / sinkhorn",
        (hc.streams, hc.sinkhorn_iters),
        (4, 20),
    );
    row(o, b, "hc eps", hc.eps, 1e-6);
    row(o, b, "collapse", hp.collapse, Collapse::Head);
    let e = hp.experts;
    row(
        o,
        b,
        "experts n / used / shared",
        (e.n_expert, e.n_used, e.n_shared),
        (256, 6, 1),
    );
    row(o, b, "experts ff", e.ff, 2048);
    row(
        o,
        b,
        "experts scale / norm",
        (e.routed_scale, e.weights_norm),
        (1.5, true),
    );
    row(o, b, "experts score", e.score, Score::SqrtSoftplus);
    row(
        o,
        b,
        "experts dense_lead / hash_layers",
        (e.dense_lead, e.hash_layers),
        (0, 3),
    );
    row(o, b, "engram", hp.engram.is_none(), true);
    row(
        o,
        b,
        "rows token_embd / engram",
        (hp.rows.token_embd, hp.rows.engram),
        (GgmlType::Q6_K, None),
    );

    let _ = writeln!(o, "layer kinds");
    set_row(o, b, &pins::WINDOW_ONLY, layers(hp, |k| !k.compressed()));
    set_row(
        o,
        b,
        &pins::SELECTED,
        layers(hp, |k| k.stream.is_some_and(|s| s.ratio == 4)),
    );
    set_row(
        o,
        b,
        &pins::DENSE,
        layers(hp, |k| k.dense.is_some_and(|d| d.ratio == 128)),
    );
    set_row(o, b, &pins::HASH_ROUTED, layers(hp, |k| k.hash_routed));
    let compressed: Vec<usize> = (2..43).collect();
    let all: Vec<usize> = (0..43).collect();
    row(o, b, "routed", layers(hp, |k| k.routed), all.clone());
    let own: Vec<usize> = hp
        .layers
        .iter()
        .enumerate()
        .filter(|&(l, k)| {
            k.stream
                .is_some_and(|s| [s.kv_source, s.index_key_source, s.topk_source] == [l; 3])
                || k.dense.is_some_and(|d| d.kv_source == l)
        })
        .map(|(l, _)| l)
        .collect();
    row(o, b, "own their stream", own, compressed.clone());
    row(
        o,
        b,
        "compressor gated + ape",
        layers(hp, |k| k.compressor.is_some_and(|c| c.gated && c.ape)),
        compressed,
    );
    row(
        o,
        b,
        "compressor overlaps",
        layers(hp, |k| k.compressor.is_some_and(|c| c.overlap)),
        pins::SELECTED.layers.to_vec(),
    );
    row(
        o,
        b,
        "index compressor gated + ape + overlap",
        layers(hp, |k| {
            k.index_compressor
                .is_some_and(|c| c.gated && c.ape && c.overlap)
        }),
        pins::SELECTED.layers.to_vec(),
    );
    row(
        o,
        b,
        "indexer",
        layers(hp, |k| k.indexer),
        pins::SELECTED.layers.to_vec(),
    );
    row(o, b, "indexer.attn_k", layers(hp, |k| k.index_keys), vec![]);
    row(
        o,
        b,
        "engram sites",
        layers(hp, |k| k.engram.is_some()),
        vec![],
    );
    row(
        o,
        b,
        "swiglu limits",
        hp.layers
            .iter()
            .all(|k| k.swiglu_limit == 10.0 && k.swiglu_limit_shared == 10.0),
        true,
    );
    row(
        o,
        b,
        "shared down type",
        layers(hp, |k| k.shared_down == GgmlType::Q8_0),
        all,
    );
    fail_if_bad(o, b);
}

/// A tensor's role family by name: finer than [`Role`], the buckets the pins
/// count bytes in.
fn family(name: &str) -> &'static str {
    let rest = name
        .strip_prefix("blk.")
        .and_then(|r| r.split_once('.'))
        .map_or(name, |(_, r)| r);
    match rest {
        r if r.starts_with("ffn_gate_exps") => "routed_gate",
        r if r.starts_with("ffn_up_exps") => "routed_up",
        r if r.starts_with("ffn_down_exps") => "routed_down",
        r if r.ends_with("_shexp.weight") => "shared_expert",
        r if r.starts_with("attn_compressor") => "compressor",
        r if r.starts_with("indexer") => "indexer",
        r if r.starts_with("attn") => "attention",
        r if r.starts_with("hc_") || r.starts_with("output_hc") => "hyper_connection",
        r if r.starts_with("ffn_gate_inp") || r.starts_with("exp_probs_b") => "router",
        r if r.starts_with("ffn_gate_tid2eid") => "hash_table",
        r if r.starts_with("ffn_norm") => "ffn_norm",
        r if r.starts_with("token_embd") => "token_embd",
        r if r.starts_with("output") => "head",
        _ => "unknown",
    }
}

#[test]
#[ignore = "needs the V4-Flash file (just gate-deepseek4-meta)"]
fn hw_deepseek4_tensor_inventory() {
    let (_, inputs) = open();
    let model = &inputs.model;
    let (mut o, mut b) = (String::new(), Vec::new());
    let (o, b) = (&mut o, &mut b);
    let total: u64 = model.tensors.iter().map(|t| t.file_bytes).sum();
    row(o, b, "tensors", model.tensors.len(), pins::TENSORS);
    row(o, b, "tensor bytes", total, pins::TENSOR_BYTES);

    let _ = writeln!(o, "bytes per family");
    let mut fam: BTreeMap<&str, (usize, u64)> = BTreeMap::new();
    for t in &model.tensors {
        let e = fam.entry(family(&t.name)).or_default();
        e.0 += 1;
        e.1 += t.file_bytes;
    }
    for &(f, n, bytes) in pins::FAMILY_BYTES {
        row(o, b, f, fam.remove(f).unwrap_or_default(), (n, bytes));
    }
    row(
        o,
        b,
        "families not pinned",
        fam.keys().copied().collect::<Vec<_>>(),
        vec![],
    );

    let _ = writeln!(o, "bytes per type");
    let mut ty: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    for t in &model.tensors {
        let e = ty
            .entry(t.ty.name().unwrap_or("?").to_string())
            .or_default();
        e.0 += 1;
        e.1 += t.file_bytes;
    }
    for &(name, n, bytes) in pins::TYPE_BYTES {
        row(o, b, name, ty.remove(name).unwrap_or_default(), (n, bytes));
    }
    row(
        o,
        b,
        "types not pinned",
        ty.keys().cloned().collect::<Vec<_>>(),
        vec![],
    );

    let _ = writeln!(o, "one expert's bytes per routed layer");
    let mut per_layer = vec![0u64; inputs.hp.n_layer];
    for t in model
        .tensors
        .iter()
        .filter(|t| t.role == Role::RoutedExperts)
    {
        let l = t.layer.expect("a routed stack has a layer");
        per_layer[l] += t.file_bytes / model.experts;
    }
    let (odd, odd_bytes) = pins::EXPERT_BYTES_MXFP4_LAYER;
    let want: Vec<u64> = (0..inputs.hp.n_layer)
        .map(|l| {
            if l == odd {
                odd_bytes
            } else {
                pins::EXPERT_BYTES
            }
        })
        .collect();
    row(o, b, "expert bytes", per_layer, want);
    row(
        o,
        b,
        "hash table role",
        model
            .tensors
            .iter()
            .filter(|t| t.role == Role::HashTable)
            .map(|t| (t.layer, t.gathered_rows))
            .collect::<Vec<_>>(),
        vec![(Some(0), Some(1)), (Some(1), Some(1)), (Some(2), Some(1))],
    );
    fail_if_bad(o, b);
}

/// The experts `card`'s budget would hold at `per_expert` bytes each, after
/// its KV, context, scratch, margin and the granules its dense uploads take
/// [derived].
fn capacity(card: &placement::Card, t: &placement::CardTotals, per_expert: u64) -> u64 {
    let used = t.kv_bytes
        + card.context_bytes
        + card.scratch_bytes
        + card.margin_bytes
        + t.dense_bytes
        + t.rounding_bytes;
    card.usable_bytes.saturating_sub(used) / per_expert
}

fn plan_rows(o: &mut String, b: &mut Vec<String>, plan: &Plan<'_>, pin: &PlanPin) {
    let _ = writeln!(o, "plan {}", pin.name);
    row(
        o,
        b,
        "violations",
        plan.violations()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        vec![],
    );
    row(o, b, "cards", plan.cards.len(), pin.cards.len());
    for ((card, t), p) in plan.machine.cards.iter().zip(&plan.cards).zip(pin.cards) {
        let p: &CardPin = p;
        row(
            o,
            b,
            &format!("{} name", p.card),
            card.name.as_str(),
            p.card,
        );
        row(o, b, &format!("{} dense", p.card), t.dense_bytes, p.dense);
        row(
            o,
            b,
            &format!("{} rounding", p.card),
            t.rounding_bytes,
            p.rounding,
        );
        row(o, b, &format!("{} kv", p.card), t.kv_bytes, p.kv);
        row(
            o,
            b,
            &format!("{} experts (planner)", p.card),
            t.experts,
            p.experts,
        );
        row(
            o,
            b,
            &format!("{} capacity [derived]", p.card),
            capacity(card, t, pins::EXPERT_BYTES),
            p.capacity,
        );
    }
    let h = &plan.host;
    row(
        o,
        b,
        "host expert bytes",
        h.expert_bytes,
        pin.host.expert_bytes,
    );
    row(
        o,
        b,
        "host table bytes",
        h.table_bytes,
        pin.host.table_bytes,
    );
    row(o, b, "host shadow bytes", h.shadow_bytes, pin.host.shadow);
    row(o, b, "host headroom", h.headroom_bytes, pin.host.headroom);
}

#[test]
#[ignore = "needs the V4-Flash file (just gate-deepseek4-meta)"]
fn hw_deepseek4_plans() {
    let (split, inputs) = open();
    let (mut o, mut b) = (String::new(), Vec::new());
    let (o, b) = (&mut o, &mut b);
    let layers = inputs.hp.n_layer;
    let ctx = workstation::CTX_MAX;
    let kv_total: u64 = (0..layers).map(|l| inputs.kv.layer_bytes(l, ctx)).sum();
    let _ = writeln!(o, "kv at ctx {ctx}: {kv_total} B over {layers} layers");
    for (pin, machine) in [
        (&pins::PLAN_A, workstation::plan_a(layers)),
        (&pins::PLAN_B, workstation::plan_b(layers)),
        (&pins::PLAN_GATE, workstation::plan_gate(layers)),
    ] {
        let plan = placement::plan_with(&inputs.model, &machine, ctx, &inputs.kv, None, None)
            .unwrap_or_else(|e| panic!("plan {}: {e}", pin.name));
        plan_rows(o, b, &plan, pin);
    }

    let _ = writeln!(o, "features the engine refuses the file for");
    let mut got: BTreeMap<String, Option<Vec<usize>>> = BTreeMap::new();
    for f in inputs.unimplemented() {
        let e = got
            .entry(f.feature.clone())
            .or_insert_with(|| f.layer.map(|_| Vec::new()));
        if let (Some(ls), Some(l)) = (e.as_mut(), f.layer) {
            ls.push(l);
        }
    }
    for ls in got.values_mut().flatten() {
        ls.sort_unstable();
    }
    let want: BTreeMap<String, Option<Vec<usize>>> = pins::UNIMPLEMENTED
        .iter()
        .map(|&(f, ls)| (f.to_string(), ls.map(<[usize]>::to_vec)))
        .collect();
    for (f, ls) in &got {
        let _ = writeln!(o, "  {f}: {ls:?}");
    }
    row(o, b, "unimplemented", got, want);
    match PlanInputs::read(&split) {
        Ok(_) => b.push("PlanInputs::read accepted the file".to_string()),
        Err(e @ placement::PlacementError::Unimplemented(_)) => {
            let _ = writeln!(o, "  PlanInputs::read: {e}");
        }
        Err(e) => b.push(format!("PlanInputs::read refused with another error: {e}")),
    }
    fail_if_bad(o, b);
}
