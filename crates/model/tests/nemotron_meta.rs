//! nemotron_h metadata gate: what the header reader resolves from the
//! Nemotron 3 Nano 4B Q4_K_M file. One contract, one test; it prints the
//! summary, then fails with every pin that differs.
//!
//! `hw_`: needs the file on the box (`just gate-nemotron-meta`). Headers
//! only: the reader refuses a layer whose tensors disagree with the
//! per-layer arrays, so the tensor headers are read and the bytes are not.

use std::fs;

use models::{Act, Ffn, Mixer, Need, RopeMode};

/// The file the gate reads.
const FILE: &str =
    "/models/NVIDIA-Nemotron-3-Nano-4B-GGUF/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf";

/// Its length, so a different file at the same path fails before the pins.
const BYTES: u64 = 2_837_072_864;

/// Mamba-2 layers: both per-layer arrays are 0.
const MAMBA: &[usize] = &[
    0, 2, 4, 6, 7, 9, 11, 14, 16, 19, 21, 23, 26, 28, 30, 31, 34, 35, 36, 38, 40,
];

/// Attention layers: `head_count_kv` 8, `feed_forward_length` 0. No rope.
const ATTN: &[usize] = &[12, 17, 24, 32];

/// Dense squared-ReLU layers.
const MLP: &[usize] = &[
    1, 3, 5, 8, 10, 13, 15, 18, 20, 22, 25, 27, 29, 33, 37, 39, 41,
];

#[test]
#[ignore = "needs the Nemotron 3 Nano 4B file on the box (just gate-nemotron-meta)"]
fn hw_nemotron_h_header() {
    let mut bad = Vec::new();
    let len = fs::metadata(FILE).map(|m| m.len());
    let split = gguf::Split::open(FILE).unwrap_or_else(|e| panic!("open {FILE}: {e}"));
    let read = model::arch::spec(&split).unwrap_or_else(|e| panic!("spec of {FILE}: {e}"));
    let s = &read.spec;
    println!(
        "nemotron_h bytes={len:?} layers={} mamba={} attn={} mlp={} hidden={} vocab={} ctx={} \
         mtp={} arch={:?}",
        s.layers.len(),
        MAMBA.len(),
        ATTN.len(),
        MLP.len(),
        s.hidden,
        s.vocab,
        s.ctx_train,
        s.mtp.len(),
        s.arch
    );
    if len.as_ref().ok() != Some(&BYTES) {
        bad.push(format!("file bytes {len:?}, want {BYTES}"));
    }
    if s.arch != models::Arch::NemotronH {
        bad.push(format!("arch {:?}", s.arch));
    }
    if s.hidden != 3136 {
        bad.push(format!("hidden {}", s.hidden));
    }
    if s.vocab != 131_072 {
        bad.push(format!("vocab {}", s.vocab));
    }
    if s.ctx_train != 1_048_576 {
        bad.push(format!("ctx {}", s.ctx_train));
    }
    if s.rms_eps.to_bits() != 0x3727_c5ac {
        bad.push(format!("rms_eps bits {:#x}", s.rms_eps.to_bits()));
    }
    if !s.mtp.is_empty() {
        bad.push(format!("mtp {}", s.mtp.len()));
    }
    if s.hc.is_some() || s.engram.is_some() {
        bad.push(format!("hc {:?} engram {:?}", s.hc, s.engram));
    }
    if s.chat.pre != "pixtral" {
        bad.push(format!("pre {}", s.chat.pre));
    }
    if s.chat.tools != Some(models::ToolFormat::QwenXml)
        || s.chat.reasoning != Some(models::ReasoningFormat::ThinkSpan)
    {
        bad.push(format!("chat tools {:?} reasoning {:?}", s.chat.tools, s.chat.reasoning));
    }
    if s.layers.len() != 42 {
        bad.push(format!("layers {}", s.layers.len()));
    }
    let mut cover = vec![0u8; s.layers.len()];
    for (name, idx) in [("mamba", MAMBA), ("attn", ATTN), ("mlp", MLP)] {
        for &l in idx {
            if l >= cover.len() {
                bad.push(format!("{name} index {l} past {}", cover.len()));
                continue;
            }
            cover[l] += 1;
            pin_layer(&mut bad, name, l, &s.layers[l]);
        }
    }
    for (l, n) in cover.iter().enumerate() {
        if *n != 1 {
            bad.push(format!("layer {l} is in {n} kind lists"));
        }
    }
    let needs = models::needs(s);
    if needs.iter().any(|(n, _)| matches!(n, Need::QkRope { .. })) {
        bad.push("an attention layer emits a rope need".to_string());
    }
    let items = model::arch::coverage::check(s, &read.tensors);
    if !items
        .iter()
        .any(|u| u.feature == "a layer program for nemotron_h")
    {
        bad.push(format!("coverage has no program item: {items:?}"));
    }
    if read.tensors.tensors.len() != 263 {
        bad.push(format!("tensors {}", read.tensors.tensors.len()));
    }
    let dims = |name: &str| {
        read.tensors
            .tensors
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.dims.clone())
    };
    if dims("blk.0.ssm_norm.weight") != Some(vec![960, 8]) {
        bad.push(format!("blk.0.ssm_norm {:?}", dims("blk.0.ssm_norm.weight")));
    }
    if dims("blk.0.ssm_in.weight") != Some(vec![3136, 17504]) {
        bad.push(format!("blk.0.ssm_in {:?}", dims("blk.0.ssm_in.weight")));
    }
    println!(
        "nemotron_h layers=42 mamba=21 attn=4 mlp=17 hidden=3136 vocab=131072 ctx=1048576 \
         rope=none mtp=0 pins_differ={}",
        bad.len()
    );
    assert!(bad.is_empty(), "{} pin(s) differ:\n  {}", bad.len(), bad.join("\n  "));
}

fn pin_layer(bad: &mut Vec<String>, name: &str, l: usize, layer: &models::LayerSpec) {
    match name {
        "mamba" => match &layer.mixer {
            Mixer::Mamba2(m) => {
                if (m.d_inner, m.d_state, m.n_heads, m.head_dim, m.n_groups, m.d_conv, m.d_in_proj)
                    != (7680, 128, 96, 80, 8, 4, 17504)
                {
                    bad.push(format!("layer {l} mamba {m:?}"));
                }
            }
            other => bad.push(format!("layer {l} mixer {other:?}, want Mamba2")),
        },
        "attn" => match &layer.mixer {
            Mixer::Gqa(g) => {
                if g.rope.mode != RopeMode::None || g.rope.dims != 0 {
                    bad.push(format!("layer {l} rope {:?}", g.rope));
                }
                if (g.heads, g.kv_heads, g.head_dim, g.value_dim) != (40, 8, 128, 128) {
                    bad.push(format!(
                        "layer {l} gqa {}/{} x {} v {}",
                        g.heads, g.kv_heads, g.head_dim, g.value_dim
                    ));
                }
            }
            other => bad.push(format!("layer {l} mixer {other:?}, want Gqa")),
        },
        "mlp" => {
            if !matches!(layer.mixer, Mixer::None) {
                bad.push(format!("layer {l} mixer {:?}, want none", layer.mixer));
            }
        }
        _ => bad.push(format!("layer {l} unknown pin {name}")),
    }
    let want_ff = if name == "mlp" { 12544 } else { 0 };
    if !matches!(
        layer.ffn,
        Ffn::Dense {
            ff,
            act: Act::ReluSqr
        } if ff == want_ff
    ) {
        bad.push(format!("layer {l} ffn {:?}, want dense {want_ff} relusqr", layer.ffn));
    }
}
