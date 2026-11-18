//! The gate fixtures of Nanbeige4.3-33B (`llama`), LFM3-1.7B (`lfm3`) and Flint Preview (`flint`):
//! each one's spec written from a synthetic source at small widths and read back, so the map's
//! refusals, the per-layer keys, the family's fills and the IQ4_XS and IQ3_S block rules (Flint's
//! routed stacks) are held without the real files. Raon-1-mini's spec (a draft beside the target)
//! has no synthetic source here: `fixture fetch raon1` runs it on the repo's public headers.

use std::fs::File;
use std::path::{Path, PathBuf};

use gguf::write::{Layout, TensorDecl, Writer};
use gguf::{GgmlType, Split, Value};
use model::arch::flint::fixture as flint;
use model::arch::lfm3::fixture as lfm3;
use model::arch::llama::fixture as nb;
use model::arch::llama::hparams::{Hparams as LlamaHparams, RopeMode};
use model::fixture::{self, CardBudget, FixtureError, FixtureSpec, Plan, Rule};

struct Dir(PathBuf);

impl Drop for Dir {
    fn drop(&mut self) {
        if let Err(e) = std::fs::remove_dir_all(&self.0)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            eprintln!("could not remove {}: {e}", self.0.display());
        }
    }
}

fn dir(tag: &str) -> Dir {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("fixture-families2-{}-{tag}", std::process::id()));
    if d.exists() {
        std::fs::remove_dir_all(&d).unwrap();
    }
    std::fs::create_dir_all(&d).unwrap();
    Dir(d)
}

type Tensors = Vec<(String, Vec<u64>, GgmlType)>;

/// A one-file source with these keys and zero tensors: the generator reads the header alone.
fn zero_source(path: &Path, kvs: &[(String, Value)], tensors: &Tensors) -> Split {
    let decls: Vec<TensorDecl> = tensors
        .iter()
        .map(|(name, dims, ty)| {
            let (_, blck, tsz) = gguf::ggml_type_info(ty.as_u32()).unwrap();
            TensorDecl {
                name: name.clone(),
                dims: dims.clone(),
                type_id: ty.as_u32(),
                nbytes: gguf::tensor_bytes(blck, tsz, dims).unwrap(),
            }
        })
        .collect();
    let bytes: Vec<u64> = decls.iter().map(|t| t.nbytes).collect();
    let layout = Layout::new(kvs, decls).unwrap();
    let mut w = Writer::new(File::create(path).unwrap(), layout).unwrap();
    for ((name, _, _), n) in tensors.iter().zip(bytes) {
        w.tensor(name, &vec![0u8; n as usize]).unwrap();
    }
    w.finish().unwrap();
    Split::open(path).unwrap()
}

fn arch_kvs(arch: &str) -> Vec<(String, Value)> {
    vec![
        (
            gguf::GENERAL_ARCHITECTURE.to_string(),
            Value::String(arch.into()),
        ),
        (
            "general.name".to_string(),
            Value::String("synthetic".into()),
        ),
    ]
}

fn layer(l: usize, stems: Vec<(&str, Vec<u64>, GgmlType)>) -> Tensors {
    stems
        .into_iter()
        .map(|(stem, dims, ty)| (format!("blk.{l}.{stem}"), dims, ty))
        .collect()
}

fn refused_with<T: std::fmt::Debug>(
    bad: &mut Vec<String>,
    what: &str,
    got: Result<T, FixtureError>,
    want: &[&str],
) {
    match got {
        Err(e) if want.iter().all(|w| e.to_string().contains(w)) => {}
        Err(e) => bad.push(format!("{what}: refused as {e}")),
        Ok(v) => bad.push(format!("{what}: accepted, {v:?}")),
    }
}

fn rule_at<'a>(p: &'a Plan, f: usize, leaf: &str) -> &'a Rule {
    let name = format!("blk.{f}.{leaf}");
    &p.target
        .tensors
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("no {name}"))
        .rule
}

/// Write the fixture of `spec` from `src` into `d/fx` and return its first shard, verified
/// against the source.
fn written(spec: &FixtureSpec, src: &Split, d: &Path, file: &str) -> Split {
    let out = d.join("fx");
    fixture::generate(spec, src, None, &out, &spec.options(), &mut |_| {}).unwrap();
    let fx = Split::open(out.join(file)).unwrap();
    assert_eq!(fx.shard_count(), 1);
    fixture::verify(spec, &fx, src, None, &mut |_, _| {}).unwrap();
    fx
}

// ------------------------------------------------------------------ Nanbeige4.3-33B (llama)

/// A synthetic `llama` source of six layers at small widths, one unsplit file: `attn_output` is
/// Q6_K on layers 0 and 1, Q8_0 on 2 and 3 and Q4_K on 4 and 5, one set or the other as the real
/// file's layers hold.
fn nb_source(path: &Path) -> Split {
    use GgmlType::{F32, Q4_K, Q6_K, Q8_0};
    const E: u64 = 256;
    const FF: u64 = 512;
    let mut tensors: Tensors = vec![
        ("token_embd.weight".into(), vec![E, 64], Q4_K),
        ("output_norm.weight".into(), vec![E], F32),
        ("output.weight".into(), vec![E, 64], Q6_K),
    ];
    for l in 0..6usize {
        let out = [Q6_K, Q8_0, Q4_K][l / 2];
        tensors.extend(layer(
            l,
            vec![
                ("attn_norm.weight", vec![E], F32),
                ("attn_q.weight", vec![E, 256], Q4_K),
                ("attn_k.weight", vec![E, 128], Q8_0),
                ("attn_v.weight", vec![E, 128], Q8_0),
                ("attn_output.weight", vec![256, E], out),
                ("ffn_norm.weight", vec![E], F32),
                ("ffn_gate.weight", vec![E, FF], Q4_K),
                ("ffn_up.weight", vec![E, FF], Q4_K),
                ("ffn_down.weight", vec![FF, E], Q4_K),
            ],
        ));
    }
    let u32v = |k: &str, v: u32| (format!("llama.{k}"), Value::U32(v));
    let mut kvs = arch_kvs("llama");
    kvs.extend([
        u32v("block_count", 6),
        u32v("context_length", 4096),
        u32v("embedding_length", E as u32),
        u32v("feed_forward_length", FF as u32),
        u32v("attention.head_count", 4),
        u32v("attention.head_count_kv", 2),
        u32v("attention.key_length", 64),
        u32v("attention.value_length", 64),
        u32v("rope.dimension_count", 64),
        u32v("vocab_size", 64),
        (
            "llama.attention.layer_norm_rms_epsilon".into(),
            Value::F32(1e-5),
        ),
        ("llama.rope.freq_base".into(), Value::F32(10_000.0)),
    ]);
    zero_source(path, &kvs, &tensors)
}

/// The production family on the synthetic source's layers: one of each `attn_output` set.
fn nb_spec() -> FixtureSpec {
    FixtureSpec {
        layers: vec![0, 2, 4],
        ..nb::spec()
    }
}

/// Contract 1: the synthetic source's fixture is written and verified (Q4_K, Q6_K, Q8_0 blocks and
/// the F32 gains), read by the engine's reader as three layers of the source's heads, with every
/// norm's gain 1; the family's budget is fixed.
#[test]
fn fixture_nanbeige_round_trip() {
    let guard = dir("nb-round-trip");
    let d = &guard.0;
    let src = nb_source(&d.join("src.gguf"));
    let spec = nb_spec();
    assert!(matches!(spec.card_budget, CardBudget::Fixed(_)));
    let plan = fixture::plan(&spec, &src, None, &spec.options()).unwrap();
    for f in 0..3 {
        for leaf in ["attn_norm.weight", "ffn_norm.weight"] {
            match rule_at(&plan, f, leaf) {
                Rule::Const { value } if *value == 1.0 => {}
                other => panic!("layer {f} {leaf}: {}, want constant 1", other.describe()),
            }
        }
    }
    let fx = written(&spec, &src, d, "llama-fixture-00001-of-00001.gguf");
    let hp = LlamaHparams::read(&fx, RopeMode::Norm).unwrap();
    assert_eq!(
        (hp.n_layer, hp.n_head, hp.n_head_kv, hp.n_ff),
        (3, 4, 2, 512)
    );
    nb::check_kinds(&spec, &fx, &src).unwrap();
}

/// Contract 2: a layer map that loses a (stem, type) pair of the source is refused by name at the
/// plan, each beside the control map, which plans.
#[test]
fn fixture_nanbeige_refuses_a_map_that_loses_a_pair() {
    let guard = dir("nb-map");
    let src = nb_source(&guard.0.join("src.gguf"));
    let mut bad = Vec::new();
    let control = nb_spec();
    if let Err(e) = fixture::plan(&control, &src, None, &control.options()) {
        bad.push(format!("the control map: {e}"));
    }
    for (what, layers, want) in [
        (
            "the Q4_K set dropped (layer 4 the first)",
            vec![0, 1, 2, 3],
            vec!["attn_output.weight", "q4_K", "source layer 4"],
        ),
        (
            "the Q6_K set dropped (layer 0 the first)",
            vec![2, 3, 4, 5],
            vec!["attn_output.weight", "q6_K", "source layer 0"],
        ),
    ] {
        let s = FixtureSpec {
            layers,
            ..nb_spec()
        };
        refused_with(
            &mut bad,
            what,
            fixture::plan(&s, &src, None, &s.options()),
            &want,
        );
    }
    assert!(bad.is_empty(), "\n  {}", bad.join("\n  "));
}

// ------------------------------------------------------------------ LFM3-1.7B (lfm3)

/// A synthetic `lfm3` source of six layers at small widths, one unsplit file: short-conv layers
/// 0, 1 and 3, sliding attention 4 and full attention 2 and 5; `ffn_down` is Q6_K on the conv
/// layers 0 and 1 and Q4_K elsewhere, `attn_v` Q6_K on the full layers and Q4_K on the sliding one,
/// so a kind and a pair each need one layer of the map.
fn lfm3_source(path: &Path) -> Split {
    use GgmlType::{F32, Q4_K, Q6_K};
    const E: u64 = 256;
    const FF: u64 = 512;
    const HEADS: u64 = 4;
    let kv: [u32; 6] = [0, 0, 2, 0, 2, 2];
    let sliding = [false, false, false, false, true, false];
    let mut tensors: Tensors = vec![
        ("token_embd.weight".into(), vec![E, 64], Q6_K),
        ("output_norm.weight".into(), vec![E], F32),
        ("rope_freqs.weight".into(), vec![32], F32),
    ];
    for l in 0..6usize {
        let down = if l < 2 { Q6_K } else { Q4_K };
        let mut stems = vec![
            ("attn_norm.weight", vec![E], F32),
            ("post_attention_norm.weight", vec![E], F32),
            ("ffn_norm.weight", vec![E], F32),
            ("post_ffw_norm.weight", vec![E], F32),
            ("ffn_gate.weight", vec![E, FF], Q4_K),
            ("ffn_up.weight", vec![E, FF], Q4_K),
            ("ffn_down.weight", vec![FF, E], down),
        ];
        if kv[l] == 0 {
            stems.extend([
                ("shortconv.in_proj.weight", vec![E, 3 * E], Q4_K),
                ("shortconv.conv.weight", vec![3, E], F32),
                ("shortconv.out_proj.weight", vec![E, E], Q4_K),
            ]);
        } else {
            let v = if sliding[l] { Q4_K } else { Q6_K };
            stems.extend([
                ("attn_q.weight", vec![E, HEADS * 64], Q4_K),
                ("attn_k.weight", vec![E, 2 * 64], Q4_K),
                ("attn_v.weight", vec![E, 2 * 64], v),
                ("attn_output.weight", vec![HEADS * 64, E], Q4_K),
                ("attn_gate.weight", vec![E, HEADS], F32),
                ("attn_q_norm.weight", vec![64], F32),
                ("attn_k_norm.weight", vec![64], F32),
            ]);
        }
        tensors.extend(layer(l, stems));
    }
    let u32v = |k: &str, v: u32| (format!("lfm3.{k}"), Value::U32(v));
    let mut kvs = arch_kvs("lfm3");
    kvs.extend([
        u32v("block_count", 6),
        u32v("context_length", 4096),
        u32v("embedding_length", E as u32),
        u32v("feed_forward_length", FF as u32),
        u32v("attention.head_count", HEADS as u32),
        (
            "lfm3.attention.head_count_kv".into(),
            Value::Array(kv.map(Value::U32).to_vec()),
        ),
        u32v("attention.key_length", 64),
        u32v("attention.value_length", 64),
        u32v("attention.sliding_window", 128),
        (
            "lfm3.attention.sliding_window_pattern".into(),
            Value::Array(sliding.map(Value::Bool).to_vec()),
        ),
        u32v("shortconv.l_cache", 3),
        u32v("rope.dimension_count", 64),
        u32v("vocab_size", 64),
        (
            "lfm3.attention.layer_norm_rms_epsilon".into(),
            Value::F32(1e-5),
        ),
        ("lfm3.rope.freq_base".into(), Value::F32(1_000_000.0)),
        ("lfm3.rope.freq_base_swa".into(), Value::F32(10_000.0)),
    ]);
    zero_source(path, &kvs, &tensors)
}

/// One layer of each kind and each pair of the synthetic source: a conv layer of the Q6_K down set,
/// a full attention layer, a conv layer of the Q4_K down set and the sliding layer.
fn lfm3_spec() -> FixtureSpec {
    FixtureSpec {
        layers: vec![0, 2, 3, 4],
        ..lfm3::spec()
    }
}

/// Contract 3: the synthetic source's fixture is written and verified (the rope factor table and the
/// gains constants 1), read by the engine's reader with the per-layer arrays at the map's layers:
/// KV heads [0, 2, 0, 2] and the pattern [false, false, false, true].
#[test]
fn fixture_lfm3_round_trip() {
    let guard = dir("lfm3-round-trip");
    let d = &guard.0;
    let src = lfm3_source(&d.join("src.gguf"));
    let spec = lfm3_spec();
    let plan = fixture::plan(&spec, &src, None, &spec.options()).unwrap();
    match plan
        .target
        .tensors
        .iter()
        .find(|t| t.name == "rope_freqs.weight")
        .map(|t| &t.rule)
    {
        Some(Rule::Const { value }) if *value == 1.0 => {}
        other => panic!("rope_freqs.weight: {other:?}, want constant 1"),
    }
    for f in 0..4 {
        match rule_at(&plan, f, "post_ffw_norm.weight") {
            Rule::Const { value } if *value == 1.0 => {}
            other => panic!("layer {f} post_ffw_norm: {}", other.describe()),
        }
    }
    let fx = written(&spec, &src, d, "lfm3-fixture-00001-of-00001.gguf");
    let hp = model::arch::lfm3::hparams::Hparams::read(&fx).unwrap();
    assert_eq!(hp.kv, [0, 2, 0, 2]);
    assert_eq!(hp.n_layer, 4);
    lfm3::check_kinds(&spec, &fx, &src).unwrap();
}

/// Contract 4: a layer map that loses a layer kind or a (stem, type) pair of the source is refused
/// by name at the plan; a written fixture read against a map of other kinds is refused by the
/// family's kinds check.
#[test]
fn fixture_lfm3_refuses_a_map_that_loses_a_kind_or_a_pair() {
    let guard = dir("lfm3-map");
    let d = &guard.0;
    let src = lfm3_source(&d.join("src.gguf"));
    let mut bad = Vec::new();
    let control = lfm3_spec();
    if let Err(e) = fixture::plan(&control, &src, None, &control.options()) {
        bad.push(format!("the control map: {e}"));
    }
    for (what, layers, want) in [
        ("no conv layer", vec![2, 4, 5], vec!["holds no Conv layer"]),
        (
            "no sliding layer",
            vec![0, 2, 3],
            vec!["holds no Sliding layer"],
        ),
        ("no full layer", vec![0, 3, 4], vec!["holds no Full layer"]),
        (
            "the Q6_K down set dropped (layer 0 the first)",
            vec![2, 3, 4, 5],
            vec!["ffn_down.weight", "q6_K", "source layer 0"],
        ),
    ] {
        let s = FixtureSpec {
            layers,
            ..lfm3_spec()
        };
        refused_with(
            &mut bad,
            what,
            fixture::plan(&s, &src, None, &s.options()),
            &want,
        );
    }
    let fx = written(&control, &src, d, "lfm3-fixture-00001-of-00001.gguf");
    // Source layer 3 is a conv layer where the fixture's layer 1 is a full attention one.
    let swapped = FixtureSpec {
        layers: vec![0, 3, 2, 4],
        ..lfm3_spec()
    };
    refused_with(
        &mut bad,
        "a swapped map",
        lfm3::check_kinds(&swapped, &fx, &src),
        &["hyperparameters"],
    );
    assert!(bad.is_empty(), "\n  {}", bad.join("\n  "));
}

// ------------------------------------------------------------------ Flint Preview (flint)

/// A synthetic `flint` source of seven layers at small widths, one unsplit file: dense layers 0 and
/// 1 (full), routed windowed layers 2, 3 and 4 (the pattern `true`) and routed full layers 5 and 6.
/// `ffn_gate_exps` and `ffn_up_exps` are IQ3_S on layer 2, IQ4_XS on 3 and 4 and Q4_K on 5 and 6,
/// `ffn_down_exps` Q6_K on 5 and Q5_K elsewhere: the IQ3_S and IQ4_XS blocks the generator
/// writes and verifies, and a pair that needs layer 2 or 3.
fn flint_source(path: &Path) -> Split {
    use GgmlType::{BF16, F32, IQ3_S, IQ4_XS, Q4_K, Q5_K, Q6_K, Q8_0};
    const E: u64 = 256;
    const HEADS: u64 = 4;
    const KV: u64 = 2;
    const X: u64 = 8;
    const XFF: u64 = 256;
    let window = [false, false, true, true, true, false, false];
    let mut tensors: Tensors = vec![
        ("token_embd.weight".into(), vec![E, 64], Q8_0),
        ("output_norm.weight".into(), vec![E], F32),
        ("output.weight".into(), vec![E, 64], Q8_0),
    ];
    for l in 0..7usize {
        let mut stems = vec![
            ("attn_norm.weight", vec![E], F32),
            ("attn_q.weight", vec![E, HEADS * 64], Q8_0),
            ("attn_k.weight", vec![E, KV * 64], Q8_0),
            ("attn_v.weight", vec![E, KV * 64], Q8_0),
            ("attn_output.weight", vec![HEADS * 64, E], Q8_0),
            ("attn_q_norm.weight", vec![64], F32),
            ("attn_k_norm.weight", vec![64], F32),
            ("attn_sinks.weight", vec![HEADS], F32),
            ("ffn_norm.weight", vec![E], F32),
        ];
        if l < 2 {
            stems.extend([
                ("ffn_gate.weight", vec![E, 512], Q8_0),
                ("ffn_up.weight", vec![E, 512], Q8_0),
                ("ffn_down.weight", vec![512, E], Q8_0),
            ]);
        } else {
            let gate_up = match l {
                2 => IQ3_S,
                3 | 4 => IQ4_XS,
                _ => Q4_K,
            };
            let down = if l == 5 { Q6_K } else { Q5_K };
            stems.extend([
                ("ffn_gate_inp.weight", vec![E, X], BF16),
                ("ffn_gate_exps.weight", vec![E, XFF, X], gate_up),
                ("ffn_up_exps.weight", vec![E, XFF, X], gate_up),
                ("ffn_down_exps.weight", vec![XFF, E, X], down),
                ("ffn_gate_shexp.weight", vec![E, XFF], Q8_0),
                ("ffn_up_shexp.weight", vec![E, XFF], Q8_0),
                ("ffn_down_shexp.weight", vec![XFF, E], Q8_0),
            ]);
        }
        tensors.extend(layer(l, stems));
    }
    let u32v = |k: &str, v: u32| (format!("flint.{k}"), Value::U32(v));
    let pattern = || Value::Array(window.map(Value::Bool).to_vec());
    let mut kvs = arch_kvs("flint");
    kvs.extend([
        u32v("block_count", 7),
        u32v("context_length", 4096),
        u32v("embedding_length", E as u32),
        u32v("feed_forward_length", 512),
        u32v("attention.head_count", HEADS as u32),
        u32v("attention.head_count_kv", KV as u32),
        u32v("attention.key_length", 64),
        u32v("attention.value_length", 64),
        u32v("attention.sliding_window", 128),
        ("flint.attention.sliding_window_pattern".into(), pattern()),
        ("flint.attention.rope_pattern".into(), pattern()),
        u32v("expert_count", X as u32),
        u32v("expert_used_count", 2),
        u32v("expert_feed_forward_length", XFF as u32),
        u32v("expert_shared_count", 1),
        u32v("expert_shared_feed_forward_length", XFF as u32),
        u32v("expert_gating_func", 2),
        ("flint.expert_weights_scale".into(), Value::F32(1.0)),
        ("flint.expert_weights_norm".into(), Value::Bool(true)),
        u32v("leading_dense_block_count", 2),
        u32v("vocab_size", 64),
        (
            "flint.attention.layer_norm_rms_epsilon".into(),
            Value::F32(1e-6),
        ),
        ("flint.rope.freq_base".into(), Value::F32(10_000.0)),
    ]);
    zero_source(path, &kvs, &tensors)
}

/// The production family on the synthetic source's layers: both dense layers, the IQ3_S window
/// layer, the IQ4_XS window layer and a full routed layer of the Q4_K and Q6_K sets, with a fixed
/// budget (the planned one needs the real header's room).
fn flint_spec() -> FixtureSpec {
    FixtureSpec {
        layers: vec![0, 1, 2, 3, 5],
        card_budget: CardBudget::Fixed(1 << 30),
        ..flint::spec()
    }
}

/// Contract 5: the synthetic source's fixture is written and verified, with the IQ3_S and IQ4_XS
/// blocks the new rules draw (their scales in the rule's band, every value the grid or kvalue
/// entry its stored bits name), read by the engine's reader with the per-layer arrays at the map's
/// layers (window [none, none, window, window, none]); the sinks are a spread, never a constant.
#[test]
fn fixture_flint_round_trip() {
    let guard = dir("flint-round-trip");
    let d = &guard.0;
    let src = flint_source(&d.join("src.gguf"));
    let spec = flint_spec();
    let plan = fixture::plan(&spec, &src, None, &spec.options()).unwrap();
    for f in 0..5 {
        match rule_at(&plan, f, "attn_sinks.weight") {
            Rule::Uniform { half_width, .. } if *half_width == flint::SINK_HALF_WIDTH => {}
            other => panic!(
                "layer {f} attn_sinks: {}, want a spread of ±{}",
                other.describe(),
                flint::SINK_HALF_WIDTH
            ),
        }
    }
    assert!(matches!(
        rule_at(&plan, 2, "ffn_gate_exps.weight"),
        Rule::Iq3S { .. }
    ));
    assert!(matches!(
        rule_at(&plan, 3, "ffn_up_exps.weight"),
        Rule::Iq4Xs { .. }
    ));
    let fx = written(&spec, &src, d, "flint-fixture-00001-of-00001.gguf");
    flint::check_kinds(&spec, &fx, &src).unwrap();
}

/// Contract 6: a layer map that loses an attention kind, a routed layer of either kind or a (stem,
/// type) pair (the IQ3_S stack, the Q6_K down) is refused by name at the plan; a written fixture
/// read against a map of other windows is refused by the family's kinds check.
#[test]
fn fixture_flint_refuses_a_map_that_loses_a_kind_or_a_pair() {
    let guard = dir("flint-map");
    let d = &guard.0;
    let src = flint_source(&d.join("src.gguf"));
    let mut bad = Vec::new();
    let control = flint_spec();
    if let Err(e) = fixture::plan(&control, &src, None, &control.options()) {
        bad.push(format!("the control map: {e}"));
    }
    for (what, layers, want) in [
        (
            "no routed full layer",
            vec![0, 1, 2, 3],
            vec!["holds no (false, true) layer"],
        ),
        (
            "no routed window layer",
            vec![0, 1, 5, 6],
            vec!["holds no (true, true) layer"],
        ),
        (
            "the IQ3_S stack dropped (layer 2)",
            vec![0, 1, 3, 4, 5],
            vec!["ffn_gate_exps.weight", "iq3_s", "source layer 2"],
        ),
        (
            "the Q6_K down dropped (layer 5)",
            vec![0, 1, 2, 3, 6],
            vec!["ffn_down_exps.weight", "q6_K", "source layer 5"],
        ),
    ] {
        let s = FixtureSpec {
            layers,
            ..flint_spec()
        };
        refused_with(
            &mut bad,
            what,
            fixture::plan(&s, &src, None, &s.options()),
            &want,
        );
    }
    let fx = written(&control, &src, d, "flint-fixture-00001-of-00001.gguf");
    // Source layer 5 is a full layer where the fixture's layer 3 is a window one.
    let swapped = FixtureSpec {
        layers: vec![0, 1, 2, 5, 3],
        ..flint_spec()
    };
    refused_with(
        &mut bad,
        "a swapped map",
        flint::check_kinds(&swapped, &fx, &src),
        &["differ"],
    );
    assert!(bad.is_empty(), "\n  {}", bad.join("\n  "));
}

/// Contract 7: the family's keys and fills as tables: the per-layer arrays follow the map, a key
/// the family does not name is refused, the norm gains are constants and the sinks a spread.
#[test]
fn the_new_families_name_their_keys_and_fills() {
    use model::fixture::KeyRule;
    assert_eq!(nb::key_rule("block_count"), Some(KeyRule::BlockCount));
    assert_eq!(nb::key_rule("rope.freq_base"), Some(KeyRule::Copy));
    assert_eq!(nb::key_rule("no_such_key"), None);
    assert_eq!(
        lfm3::key_rule("attention.head_count_kv"),
        Some(KeyRule::PerLayer)
    );
    assert_eq!(
        lfm3::key_rule("attention.sliding_window_pattern"),
        Some(KeyRule::PerLayer)
    );
    assert_eq!(lfm3::const_value("rope_freqs.weight"), Some(1.0));
    for key in ["attention.sliding_window_pattern", "attention.rope_pattern"] {
        assert_eq!(flint::key_rule(key), Some(KeyRule::PerLayer), "{key}");
    }
    assert_eq!(
        flint::spread_value("attn_sinks.weight"),
        Some(flint::SINK_HALF_WIDTH)
    );
    assert_eq!(flint::const_value("attn_sinks.weight"), None);
    assert_eq!(flint::const_value("ffn_gate_inp.weight"), None);
    assert_eq!(flint::key_rule("no_such_key"), None);
}
