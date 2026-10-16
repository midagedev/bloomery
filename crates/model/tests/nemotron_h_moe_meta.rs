//! nemotron_h_moe metadata gate: what the header reader resolves from the
//! Nemotron 3.5 Bolt 74B-A9B 24GB GGUF
//! (`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF`). One contract, one
//! test; it prints the summary, then fails with every pin that differs.
//!
//! `hw_`: needs the file on the box at [`FILE`] (`just
//! gate-nemotron-h-moe-meta`). Headers only: the reader
//! holds every tensor to the dims the keys give it, so the tensor headers are
//! read and the bytes are not. The tensors are pinned to tile the file's data
//! to its last byte, so a block size the reader gets wrong fails a pin.

use std::collections::BTreeMap;
use std::fs;

use gguf::GgmlType;
use models::{Act, Ffn, Mixer, RopeMode, Score, ToolFormat};

/// The box's path of the file.
const FILE: &str = "/models/nemotron-3.5-bolt/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf";

/// Its length (the Hugging Face listing), where the last tensor ends, so a
/// different file at the same path fails before the pins.
const BYTES: u64 = 22_018_326_528;

/// Attention layers: `head_count_kv` 2, `feed_forward_length` 0. No rope.
const ATTN: &[usize] = &[7, 16, 25, 36, 47, 58, 69, 78];

/// The (expert count, expert width) pairs of the MoE layers, each with the
/// layers that carry it.
const SHAPES: &[((u32, u32), usize)] = &[
    ((240, 1536), 1),
    ((240, 2048), 7),
    ((240, 2304), 4),
    ((304, 2048), 2),
    ((304, 2304), 3),
    ((304, 2560), 1),
    ((368, 2048), 2),
    ((368, 2304), 4),
    ((368, 2560), 3),
    ((432, 2304), 4),
    ((432, 2560), 6),
    ((496, 2560), 3),
];

/// Each tensor family (the name past `blk.N.`) and its type.
const FAMILIES: &[(&str, GgmlType)] = &[
    ("token_embd.weight", GgmlType::Q8_0),
    ("output.weight", GgmlType::Q8_0),
    ("output_norm.weight", GgmlType::F32),
    ("attn_norm.weight", GgmlType::F32),
    ("ssm_in.weight", GgmlType::NVFP4),
    ("ssm_out.weight", GgmlType::NVFP4),
    ("ssm_in.scale", GgmlType::F32),
    ("ssm_out.scale", GgmlType::F32),
    ("ssm_a", GgmlType::F32),
    ("ssm_d", GgmlType::F32),
    ("ssm_conv1d.weight", GgmlType::F32),
    ("ssm_conv1d.bias", GgmlType::F32),
    ("ssm_dt.bias", GgmlType::F32),
    ("ssm_norm.weight", GgmlType::F32),
    ("attn_q.weight", GgmlType::Q8_0),
    ("attn_k.weight", GgmlType::Q8_0),
    ("attn_v.weight", GgmlType::Q8_0),
    ("attn_output.weight", GgmlType::Q8_0),
    ("ffn_gate_inp.weight", GgmlType::F32),
    ("exp_probs_b.bias", GgmlType::F32),
    ("ffn_latent_down.weight", GgmlType::Q8_0),
    ("ffn_latent_up.weight", GgmlType::Q8_0),
    ("ffn_up_exps.weight", GgmlType::IQ2_XXS),
    ("ffn_down_exps.weight", GgmlType::IQ2_XXS),
    ("ffn_up_shexp.weight", GgmlType::IQ2_XXS),
    ("ffn_down_shexp.weight", GgmlType::IQ2_XXS),
];

/// The file's bytes by type, from the tensor headers.
const TYPE_BYTES: &[(GgmlType, u64)] = &[
    (GgmlType::F32, 237_570_368),
    (GgmlType::Q8_0, 1_800_404_992),
    (GgmlType::IQ2_XXS, 17_506_959_360),
    (GgmlType::NVFP4, 2_465_464_320),
];

/// The coverage list, feature by feature in the order the refusal prints
/// them, each with how many layers carry it (0: the whole model): empty, as
/// `Program::NemotronHMoeBody` runs every part of the file — its routers at
/// the five widths, the latent experts, the group-16 flash, the NVFP4 and
/// IQ2_XXS sites and the per-layer stack shapes.
const COVERAGE: &[(&str, usize)] = &[];

#[test]
#[ignore = "needs the Bolt 24GB GGUF on the box (just gate-nemotron-h-moe-meta)"]
fn hw_nemotron_h_moe_header() {
    let file = FILE;
    let len = fs::metadata(file).map(|m| m.len());
    let mut bad = Vec::new();
    let split = gguf::Split::open(file).unwrap_or_else(|e| panic!("open {file}: {e}"));
    let read = model::arch::spec(&split).unwrap_or_else(|e| panic!("spec: {e}"));
    let hp = model::arch::nemotron_h_moe::hparams::Hparams::read(&split)
        .unwrap_or_else(|e| panic!("hparams: {e}"));
    let unread = model::arch::nemotron_h_moe::hparams::unread_keys(&split);
    let s = &read.spec;
    let g = split.shard(0).expect("one shard");
    let items = model::arch::coverage::check(s, &read.tensors);
    println!(
        "nemotron_h_moe file={file} bytes={len:?} kvs={} tensors={} layers={} hidden={} vocab={} \
         ctx={} defaults={:?} unread={unread:?}",
        g.kv_count(),
        split.tensor_count(),
        s.layers.len(),
        s.hidden,
        s.vocab,
        s.ctx_train,
        read.defaults,
    );
    println!(
        "coverage: {}",
        model::placement::PlacementError::Unimplemented(items.clone())
    );

    // The tensors tile the data section: each starts where the one before
    // it ends, rounded up to the alignment, and the last ends at the file's
    // length. A tensor sized at another block size breaks the tiling.
    let mut by_offset: Vec<_> = g.iter_tensors().collect();
    by_offset.sort_by_key(|t| t.offset);
    let mut end = 0u64;
    for t in &by_offset {
        if t.offset != end.next_multiple_of(g.alignment()) {
            bad.push(format!(
                "{} at {}, the previous ends at {end}",
                t.name, t.offset
            ));
        }
        end = t.offset + t.nbytes;
    }
    if len.as_ref().ok() != Some(&BYTES) {
        bad.push(format!("file bytes {len:?}, want {BYTES}"));
    }
    if g.data_base() + end != BYTES {
        bad.push(format!(
            "the data ends at {}, want {BYTES}",
            g.data_base() + end
        ));
    }
    if (g.kv_count(), split.tensor_count()) != (51, 843) {
        bad.push(format!(
            "{} keys and {} tensors",
            g.kv_count(),
            split.tensor_count()
        ));
    }

    if s.arch != models::Arch::NemotronHMoe {
        bad.push(format!("arch {:?}", s.arch));
    }
    if (s.hidden, s.vocab, s.ctx_train) != (4096, 131_072, 262_144) {
        bad.push(format!(
            "hidden {} vocab {} ctx {}",
            s.hidden, s.vocab, s.ctx_train
        ));
    }
    if s.rms_eps.to_bits() != 0x3727_c5ac {
        bad.push(format!("rms_eps bits {:#x}", s.rms_eps.to_bits()));
    }
    if !s.mtp.is_empty() || s.hc.is_some() || s.engram.is_some() || s.encoder.is_some() {
        bad.push("mtp, hc, engram or encoder".to_string());
    }
    if s.chat.pre != "pixtral"
        || s.chat.tools != Some(ToolFormat::QwenXml)
        || s.chat.reasoning != Some(models::ReasoningFormat::ThinkSpan)
    {
        bad.push(format!(
            "chat pre {} tools {:?} reasoning {:?}",
            s.chat.pre, s.chat.tools, s.chat.reasoning
        ));
    }
    if !read.defaults.is_empty() {
        bad.push(format!("defaults {:?}", read.defaults));
    }
    let want_unread = [
        "general.sampling.top_p",
        "general.sampling.temp",
        "nemotron_h_moe.rope.scaling.finetuned",
    ];
    if unread != want_unread {
        bad.push(format!("unread keys {unread:?}"));
    }

    // 88 layers: 40 Mamba-2, 40 MoE, 8 attention at ATTN.
    let (mut mamba, mut moe, mut attn) = (Vec::new(), Vec::new(), Vec::new());
    let mut shapes: BTreeMap<(u32, u32), usize> = BTreeMap::new();
    let mut routed = 0u32;
    for (l, layer) in s.layers.iter().enumerate() {
        match (&layer.mixer, &layer.ffn) {
            (Mixer::Mamba2(m), Ffn::Dense { ff: 0, .. }) => {
                mamba.push(l);
                let got = (
                    m.d_inner,
                    m.n_heads,
                    m.head_dim,
                    m.n_groups,
                    m.d_state,
                    m.d_conv,
                    m.d_in_proj,
                );
                if got != (8192, 128, 64, 8, 128, 4, 18_560) {
                    bad.push(format!("layer {l} Mamba-2 {got:?}"));
                }
            }
            (Mixer::Gqa(q), Ffn::Dense { ff: 0, .. }) => {
                attn.push(l);
                let got = (q.heads, q.kv_heads, q.group(), q.head_dim, q.value_dim);
                if got != (32, 2, 16, 128, 128)
                    || q.rope.mode != RopeMode::None
                    || q.qk_norm
                    || q.window.is_some()
                {
                    bad.push(format!("layer {l} attention {q:?}"));
                }
            }
            (Mixer::None, Ffn::Moe(m)) => {
                moe.push(l);
                *shapes.entry((m.experts, m.expert_ff)).or_default() += 1;
                routed += m.experts;
                let sh = m.shared.expect("a shared expert");
                if m.top_k != 9
                    || m.latent != Some(1024)
                    || m.act != Act::ReluSqr
                    || m.router.score != Score::Sigmoid
                    || !m.router.bias
                    || !m.router.norm
                    || m.router.scale != 5.0
                    || m.router.hash
                    || (sh.ff, sh.act, sh.sigmoid_gate) != (5376, Act::ReluSqr, false)
                {
                    bad.push(format!("layer {l} MoE {m:?}"));
                }
            }
            other => bad.push(format!("layer {l}: {other:?}")),
        }
    }
    if s.layers.len() != 88 || (mamba.len(), moe.len()) != (40, 40) || attn != ATTN {
        bad.push(format!(
            "{} layers: {} Mamba-2, {} MoE, attention at {attn:?}",
            s.layers.len(),
            mamba.len(),
            moe.len()
        ));
    }
    let want_shapes: BTreeMap<(u32, u32), usize> = SHAPES.iter().copied().collect();
    if shapes != want_shapes {
        bad.push(format!("expert shapes {shapes:?}"));
    }
    if routed != 13_824 || hp.n_expert != 496 {
        bad.push(format!(
            "{routed} routed experts, expert_count {}",
            hp.n_expert
        ));
    }
    let t = &read.tensors;
    if (t.layers, t.experts, t.experts_used) != (88, 496, 9) {
        bad.push(format!(
            "tensors' counts {} {} {}",
            t.layers, t.experts, t.experts_used
        ));
    }

    // The quant mix: every family at its one type, and the bytes by type.
    let mut types: BTreeMap<&str, Vec<GgmlType>> = BTreeMap::new();
    let mut bytes: BTreeMap<GgmlType, u64> = BTreeMap::new();
    for (_, info) in split.iter_tensors() {
        let family = match info.name.strip_prefix("blk.") {
            Some(rest) => rest.split_once('.').map_or(rest, |(_, stem)| stem),
            None => info.name.as_str(),
        };
        let seen = types.entry(family).or_default();
        if !seen.contains(&info.ty) {
            seen.push(info.ty);
        }
        *bytes.entry(info.ty).or_default() += info.nbytes;
    }
    let want_types: BTreeMap<&str, Vec<GgmlType>> =
        FAMILIES.iter().map(|&(f, ty)| (f, vec![ty])).collect();
    if types != want_types {
        bad.push(format!("families {types:?}"));
    }
    let want_bytes: BTreeMap<GgmlType, u64> = TYPE_BYTES.iter().copied().collect();
    if bytes != want_bytes {
        bad.push(format!("bytes by type {bytes:?}"));
    }

    // The coverage list, as the refusal prints it.
    let mut features: Vec<(&str, usize)> = Vec::new();
    for u in &items {
        match features.iter_mut().find(|(f, _)| *f == u.feature) {
            Some((_, n)) => *n += usize::from(u.layer.is_some()),
            None => features.push((&u.feature, usize::from(u.layer.is_some()))),
        }
    }
    if features != COVERAGE {
        bad.push(format!("coverage {features:#?}"));
    }

    assert!(
        bad.is_empty(),
        "{} pin(s) differ:\n{}",
        bad.len(),
        bad.join("\n")
    );
}
