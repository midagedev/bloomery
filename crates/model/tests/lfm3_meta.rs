//! lfm3 metadata gate: what the header reader resolves from the LFM3-1.7B Q4_K_M file. One
//! contract, one test; it prints the summary, then fails with every pin that differs. The second
//! test holds the rope's factor table to the shape its description assumes.
//!
//! `hw_`: needs the file on the box (`just gate-lfm3-meta`). The reader refuses a layer whose
//! tensors disagree with the per-layer arrays, so the tensor headers are read; the bytes are not,
//! but the rope table's 64 values.

use std::fs;

use models::{Act, Ffn, Mixer, Need, OutGate, ReasoningFormat, Residual, RopeMode, ToolFormat};

/// The file the gate reads.
const FILE: &str = "/models/LFM3-1.7B-GGUF/LFM3-1.7B-Q4_K_M.gguf";

/// Its length, so a different file at the same path fails before the pins.
const BYTES: u64 = 1_099_627_808;

/// Short-conv layers: `attention.head_count_kv` is 0.
const CONV: &[usize] = &[0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23];

/// Sliding-window attention layers: `head_count_kv` 4, the pattern true.
const SLIDING: &[usize] = &[4, 8, 12, 16, 20];

/// Full attention layers: `head_count_kv` 4, the pattern false.
const FULL: &[usize] = &[2, 6, 10, 14, 18, 22];

#[test]
#[ignore = "needs the LFM3-1.7B Q4_K_M file on the box (just gate-lfm3-meta)"]
fn hw_lfm3_header() {
    let mut bad = Vec::new();
    let len = fs::metadata(FILE).map(|m| m.len());
    let split = gguf::Split::open(FILE).unwrap_or_else(|e| panic!("open {FILE}: {e}"));
    let read = model::arch::spec(&split).unwrap_or_else(|e| panic!("spec of {FILE}: {e}"));
    let s = &read.spec;
    println!(
        "lfm3 bytes={len:?} layers={} conv={} sliding={} full={} hidden={} vocab={} ctx={} mtp={} \
         arch={:?}",
        s.layers.len(),
        CONV.len(),
        SLIDING.len(),
        FULL.len(),
        s.hidden,
        s.vocab,
        s.ctx_train,
        s.mtp.len(),
        s.arch
    );
    if len.as_ref().ok() != Some(&BYTES) {
        bad.push(format!("file bytes {len:?}, want {BYTES}"));
    }
    if s.arch != models::Arch::Lfm3 {
        bad.push(format!("arch {:?}", s.arch));
    }
    if s.hidden != 2048 {
        bad.push(format!("hidden {}", s.hidden));
    }
    if s.vocab != 128_000 {
        bad.push(format!("vocab {}", s.vocab));
    }
    if s.ctx_train != 262_144 {
        bad.push(format!("ctx {}", s.ctx_train));
    }
    if s.rms_eps.to_bits() != 0x3727_c5ac {
        bad.push(format!("rms_eps bits {:#x}", s.rms_eps.to_bits()));
    }
    if !s.mtp.is_empty() {
        bad.push(format!("mtp {}", s.mtp.len()));
    }
    if s.hc.is_some() || s.engram.is_some() || s.encoder.is_some() || s.logit_epilogue.is_some() {
        bad.push(format!(
            "hc {:?} engram {:?} encoder {:?} logit epilogue {:?}",
            s.hc, s.engram, s.encoder, s.logit_epilogue
        ));
    }
    if s.chat.pre != "lfm2" {
        bad.push(format!("pre {}", s.chat.pre));
    }
    if s.chat.tools != Some(ToolFormat::QwenXml) || s.chat.reasoning != Some(ReasoningFormat::THINK)
    {
        bad.push(format!(
            "chat tools {:?} reasoning {:?}",
            s.chat.tools, s.chat.reasoning
        ));
    }
    // The template teaches the markup the description names, not a format of its own.
    match &s.chat.template {
        Some(t) => {
            if ToolFormat::of_template(t) != Some(ToolFormat::QwenXml)
                || ReasoningFormat::of_template(t) != Some(ReasoningFormat::THINK)
            {
                bad.push(format!(
                    "template teaches tools {:?} reasoning {:?}",
                    ToolFormat::of_template(t),
                    ReasoningFormat::of_template(t)
                ));
            }
        }
        None => bad.push("no chat template".to_string()),
    }
    if !tokenizer::runs_pre_tokenizer(&s.chat.pre) {
        bad.push(format!("the tokenizer does not run pre {}", s.chat.pre));
    }
    if s.layers.len() != 24 {
        bad.push(format!("layers {}", s.layers.len()));
    }
    let mut cover = vec![0u8; s.layers.len()];
    for (name, idx) in [("conv", CONV), ("sliding", SLIDING), ("full", FULL)] {
        for &l in idx {
            if l >= cover.len() {
                bad.push(format!("{name} index {l} past {}", cover.len()));
                continue;
            }
            cover[l] += 1;
            pin_layer(&mut bad, name, l, &s.layers[l], s.rms_eps);
        }
    }
    for (l, n) in cover.iter().enumerate() {
        if *n != 1 {
            bad.push(format!("layer {l} is in {n} kind lists"));
        }
    }
    // No program runs the description: the coverage names each part no body runs, by layer.
    let items = model::arch::coverage::check(s, &read.tensors);
    println!("coverage items: {}", items.len());
    for u in &items {
        println!("  {:?}: {}", u.layer, u.feature);
    }
    let named = |layer: usize, feature: &str| {
        items
            .iter()
            .any(|u| u.layer == Some(layer) && u.feature == feature)
    };
    for &l in CONV {
        if !named(l, &Need::ShortConv { taps: 7 }.to_string()) {
            bad.push(format!(
                "coverage does not name the short conv of layer {l}"
            ));
        }
    }
    for &l in SLIDING.iter().chain(FULL) {
        if !named(l, &Need::OutGateProjHead.to_string()) {
            bad.push(format!("coverage does not name the head gate of layer {l}"));
        }
    }
    for &l in FULL {
        if !named(l, &Need::RopeFactors.to_string()) {
            bad.push(format!(
                "coverage does not name the rope factors of layer {l}"
            ));
        }
    }
    for &l in SLIDING {
        if named(l, &Need::RopeFactors.to_string()) {
            bad.push(format!("coverage names rope factors on sliding layer {l}"));
        }
    }
    if !items
        .iter()
        .any(|u| u.layer.is_none() && u.feature == "a layer program for lfm3")
    {
        bad.push("coverage does not name the missing program".to_string());
    }
    if items
        .iter()
        .any(|u| u.feature.starts_with("pre-tokenizer") || u.feature.contains("tool-call parser"))
    {
        bad.push("coverage names the pre-tokenizer or the tool parser".to_string());
    }
    let tensors = &read.tensors;
    if tensors.tensors.len() != 287 {
        bad.push(format!("tensors {}", tensors.tensors.len()));
    }
    if (tensors.layers, tensors.experts, tensors.experts_used) != (24, 0, 0) {
        bad.push(format!(
            "counts {} {} {}",
            tensors.layers, tensors.experts, tensors.experts_used
        ));
    }
    let find = |name: &str| tensors.tensors.iter().find(|t| t.name == name);
    for (name, dims) in [
        ("token_embd.weight", vec![2048, 128_000]),
        ("output_norm.weight", vec![2048]),
        ("rope_freqs.weight", vec![64]),
        ("blk.0.shortconv.in_proj.weight", vec![2048, 6144]),
        ("blk.0.shortconv.conv.weight", vec![7, 2048]),
        ("blk.0.shortconv.out_proj.weight", vec![2048, 2048]),
        ("blk.2.attn_q.weight", vec![2048, 2048]),
        ("blk.2.attn_k.weight", vec![2048, 512]),
        ("blk.2.attn_gate.weight", vec![2048, 16]),
        ("blk.2.attn_q_norm.weight", vec![128]),
        ("blk.2.ffn_gate.weight", vec![2048, 7680]),
        ("blk.2.ffn_down.weight", vec![7680, 2048]),
    ] {
        match find(name) {
            Some(t) if t.dims == dims => {}
            other => bad.push(format!(
                "{name} {:?}, want {dims:?}",
                other.map(|t| &t.dims)
            )),
        }
    }
    // The head is the token embedding: the file carries no `output.weight`.
    if find("output.weight").is_some() {
        bad.push("output.weight is in the file".to_string());
    }
    if find("rope_freqs.weight").map(|t| (t.role, t.layer))
        != Some((model::placement::Role::Unread, None))
    {
        bad.push("rope_freqs.weight is not the load's unread table".to_string());
    }
    println!(
        "lfm3 layers=24 conv=13 sliding=5 full=6 taps=7 window=256 hidden=2048 vocab=128000 \
         ctx=262144 head=tied pins_differ={}",
        bad.len()
    );
    assert!(
        bad.is_empty(),
        "{} pin(s) differ:\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}

/// The rope's factor table is 32 factors of 1 and then 32 of 1e30: the first 32 pairs of a full
/// layer's head turn, and the last 32 are the identity. The description carries only that the
/// layer has factors and the load reads the table itself, so the table's shape is this gate's.
#[test]
#[ignore = "needs the LFM3-1.7B Q4_K_M file on the box (just gate-lfm3-meta)"]
fn hw_lfm3_rope_factors() {
    let split = gguf::Split::open(FILE).unwrap_or_else(|e| panic!("open {FILE}: {e}"));
    let bytes = split
        .data("rope_freqs.weight")
        .unwrap_or_else(|e| panic!("rope_freqs.weight: {e}"));
    assert_eq!(bytes.len(), 64 * 4, "the table's bytes");
    let factors: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let (turn, still) = factors.split_at(32);
    assert!(
        turn.iter().all(|&f| f == 1.0),
        "the first 32 factors: {turn:?}"
    );
    assert!(
        still.iter().all(|&f| f == 1e30_f32),
        "the last 32 factors: {still:?}"
    );
    println!("lfm3 rope factors: 32 x 1.0, then 32 x 1e30");
}

/// Layer `l`'s mixer, feed-forward block and residual against the pins of its kind.
fn pin_layer(bad: &mut Vec<String>, name: &str, l: usize, layer: &models::LayerSpec, rms_eps: f32) {
    match (name, &layer.mixer) {
        ("conv", Mixer::ShortConv(c)) => {
            if c.taps != 7 {
                bad.push(format!("layer {l} taps {}", c.taps));
            }
        }
        ("sliding" | "full", Mixer::Gqa(g)) => {
            if (g.heads, g.kv_heads, g.head_dim, g.value_dim) != (16, 4, 128, 128) {
                bad.push(format!(
                    "layer {l} gqa {}/{} x {} v {}",
                    g.heads, g.kv_heads, g.head_dim, g.value_dim
                ));
            }
            if !g.qk_norm || g.o_norm || g.sinks || g.select.is_some() || g.value_scale.is_some() {
                bad.push(format!("layer {l} norms or selector {g:?}"));
            }
            if g.out_gate != OutGate::ProjHead {
                bad.push(format!("layer {l} gate {:?}", g.out_gate));
            }
            if g.rope.mode != RopeMode::Neox || g.rope.dims != 128 || g.rope.yarn.is_some() {
                bad.push(format!("layer {l} rope {:?}", g.rope));
            }
            let (base, factors, window) = if name == "sliding" {
                (10_000.0, false, Some(256))
            } else {
                (1_000_000.0, true, None)
            };
            if g.rope.base != base || g.rope.factors != factors || g.window != window {
                bad.push(format!(
                    "layer {l} base {} factors {} window {:?}, want {base} {factors} {window:?}",
                    g.rope.base, g.rope.factors, g.window
                ));
            }
        }
        (_, other) => bad.push(format!("layer {l} mixer {other:?}, want {name}")),
    }
    if !matches!(
        layer.ffn,
        Ffn::Dense {
            ff: 7680,
            act: Act::SwiGlu { limit: None }
        }
    ) {
        bad.push(format!(
            "layer {l} ffn {:?}, want dense 7680 swiglu",
            layer.ffn
        ));
    }
    if layer.residual
        != (Residual::Sandwich {
            eps: rms_eps,
            add: [1.0, 1.0],
        })
    {
        bad.push(format!("layer {l} residual {:?}", layer.residual));
    }
    if !layer.extras.is_empty() {
        bad.push(format!("layer {l} extras {:?}", layer.extras));
    }
}
