//! exaone-moe metadata gate: what the header reader resolves from the fixture
//! GGUF — `bloomery-quantize --hf`'s bootstrap mix of the cut folder
//! `/models/fixtures/exaoneh` (layers 0, 1, 2, 5, 62 and 65 of the real model
//! renumbered 0 to 5, every per-layer array cut alike, the MTP block kept),
//! the one file both engines read. The gate pins the file's identity (the
//! header prefix's bytes and sha256), every architecture key, the description
//! layer by layer (the window/rope/act table, the biased sigmoid router, the
//! MTP block in `spec.mtp`), the plan's own values (the one routed width, the
//! one rope, the one router instance, the store rows each layer's window
//! counts) and every tensor's name, role, type and shape; and the reader's
//! type admission (r6b's finding): a tensor in a type the architecture's stem
//! table does not hold its stem in — the Q4_K `nextn.eh_proj` the quantizer's
//! first rules wrote — is refused at the read by name. Headers only: the
//! 8.5 GB payload is not read.

use gguf::write::{Layout, TensorDecl};
use gguf::{GENERAL_ARCHITECTURE, GgmlType, Split, Value};
use models::{Act, Ffn, Mixer, Need, RopeMode, RouterNorm};

/// The box's path of the fixture GGUF.
const FILE: &str = "/models/fixtures/exaoneh-gguf/exaoneh-gguf-fixture-00001-of-00001.gguf";

/// The file's whole length, so a different file at the same path fails before
/// the pins.
const BYTES: u64 = 8_496_145_440;

/// The header's bytes and its sha256: the description's identity.
const HEADER_BYTES: u64 = 6_551_584;
const HEADER_SHA: &str = "b78a549650d574ac86faeda3a48677ae78eab3e971697e2c7b06e02ff00978d2";

/// The file's embedded chat template, in bytes: the model's own, whose markup
/// selects the QwenXml tool parser and the THINK reasoning span.
const TEMPLATE: usize = 6_647;

/// The trunk's six layers and the next-token one.
const TRUNK: usize = 6;
const BLOCKS: usize = 7;

/// The fixture's per-layer facts, by layer: the sliding window (0: full
/// attention), whether the layer's head turns and the SwiGLU limit its routed
/// and shared experts take (the cut's layers 4 and 5 are the real model's 62
/// and 65, the clamped ones).
const LAYERS: [(u32, bool, f32); BLOCKS] = [
    (0, false, 0.0),
    (4096, true, 0.0),
    (128, true, 0.0),
    (0, false, 0.0),
    (128, true, 7.0),
    (0, false, 7.0),
    (128, true, 0.0),
];

/// The store rows a layer's window counts at a context of 8192: the full
/// layers the context, layer 1's 4096-window ring 4608 rows, the 128-windows'
/// 768 (`ring_rows`: the window plus one prompt batch less one, rounded to a
/// 256-row granule, never past the context).
const ROWS_AT_8192: [u64; TRUNK] = [8192, 4608, 768, 8192, 768, 8192];

/// The file's inventory as the declaration list a strict header-only `Split`
/// lays out, with `edit` over the declarations.
fn header_of(edit: impl Fn(&mut Vec<TensorDecl>)) -> Split {
    let inv = gguf::inventory_of(FILE).expect("the GGUF header parses");
    let mut decls: Vec<TensorDecl> = inv
        .tensors
        .iter()
        .map(|t| TensorDecl {
            name: t.name.clone(),
            dims: t.dims.clone(),
            type_id: t.type_id,
            nbytes: t.nbytes.expect("every type of the file has a known size"),
        })
        .collect();
    edit(&mut decls);
    let layout = Layout::new(&inv.meta, decls).expect("strict tensor layout");
    let path =
        std::env::temp_dir().join(format!("bloomery-exaone-meta-{}.gguf", std::process::id()));
    Split::header_only(path, &layout).expect("strict header-only Split")
}

/// One tensor's declaration retyped: its bytes recomputed for the type.
fn retype(decls: &mut [TensorDecl], name: &str, ty: GgmlType) {
    let t = decls
        .iter_mut()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("{name} is in the file"));
    t.type_id = ty.as_u32();
    let (_, blck, tsz) = gguf::ggml_type_info(ty.as_u32()).expect("a type");
    t.nbytes = gguf::tensor_bytes(blck, tsz, &t.dims).expect("bytes");
}

/// What the reader resolves from the fixture GGUF's own header, and what it
/// refuses: every pin that differs is listed, then the test fails.
#[test]
#[ignore = "needs the fixture GGUF on the box (just gate-exaone-meta)"]
fn hw_exaone_fixture_header() {
    let path = std::path::PathBuf::from(FILE);
    let inv = gguf::inventory_of(&path).expect("the GGUF header parses");
    let mut bad = Vec::new();
    let file_len = std::fs::metadata(&path).expect("the file's length").len();
    if file_len != BYTES {
        bad.push(format!(
            "the file is {file_len} B, the fixture GGUF is {BYTES}"
        ));
    }
    if inv.data_base != HEADER_BYTES {
        bad.push(format!("data_base {} bytes", inv.data_base));
    }
    let mut file = std::fs::File::open(&path).unwrap();
    let mut prefix = std::io::Read::take(&mut file, HEADER_BYTES);
    let sha = gguf::digest::sha256_reader(&path, &mut prefix).unwrap();
    if sha != HEADER_SHA {
        bad.push(format!("header sha256 {sha}"));
    }
    if inv.tensors.len() != 104 {
        bad.push(format!("{} tensors, the file holds 104", inv.tensors.len()));
    }
    let kv: std::collections::BTreeMap<&str, &Value> =
        inv.meta.iter().map(|(k, v)| (k.as_str(), v)).collect();
    for (key, want) in [
        (GENERAL_ARCHITECTURE, Value::String("exaone-moe".into())),
        ("exaone-moe.vocab_size", Value::U32(153_600)),
        ("exaone-moe.context_length", Value::U32(262_144)),
        ("exaone-moe.embedding_length", Value::U32(6144)),
        ("exaone-moe.block_count", Value::U32(BLOCKS as u32)),
        ("exaone-moe.feed_forward_length", Value::U32(1024)),
        ("exaone-moe.leading_dense_block_count", Value::U32(2)),
        ("exaone-moe.nextn_predict_layers", Value::U32(1)),
        ("exaone-moe.attention.head_count", Value::U32(64)),
        ("exaone-moe.attention.head_count_kv", Value::U32(8)),
        ("exaone-moe.attention.key_length", Value::U32(128)),
        ("exaone-moe.attention.value_length", Value::U32(128)),
        (
            "exaone-moe.attention.layer_norm_rms_epsilon",
            Value::F32(1e-5),
        ),
        ("exaone-moe.rope.freq_base", Value::F32(1e6)),
        ("exaone-moe.rope.dimension_count", Value::U32(128)),
        ("exaone-moe.expert_count", Value::U32(256)),
        ("exaone-moe.expert_used_count", Value::U32(8)),
        ("exaone-moe.expert_shared_count", Value::U32(1)),
        ("exaone-moe.expert_feed_forward_length", Value::U32(512)),
        (
            "exaone-moe.expert_shared_feed_forward_length",
            Value::U32(512),
        ),
        ("exaone-moe.expert_gating_func", Value::U32(2)),
        ("exaone-moe.expert_weights_norm", Value::Bool(true)),
        ("exaone-moe.expert_weights_scale", Value::F32(2.5)),
        ("exaone-moe.expert_group_count", Value::U32(1)),
        ("exaone-moe.expert_group_used_count", Value::U32(1)),
        ("tokenizer.ggml.model", Value::String("gpt2".into())),
        ("tokenizer.ggml.pre", Value::String("exaone-moe".into())),
        ("tokenizer.ggml.normalizer", Value::String("NFC".into())),
        ("tokenizer.ggml.bos_token_id", Value::U32(1)),
        ("tokenizer.ggml.eos_token_id", Value::U32(53)),
        ("tokenizer.ggml.unknown_token_id", Value::U32(3)),
        ("tokenizer.ggml.padding_token_id", Value::U32(0)),
        ("tokenizer.ggml.add_bos_token", Value::Bool(false)),
        ("tokenizer.ggml.add_eos_token", Value::Bool(false)),
        (
            "tokenizer.bloomery.eog_token_ids",
            Value::Array(vec![Value::U32(2), Value::U32(53)]),
        ),
    ] {
        if kv.get(key) != Some(&&want) {
            bad.push(format!("{key} is {:?}", kv.get(key)));
        }
    }
    let windows: Vec<i32> = LAYERS.iter().map(|&(w, _, _)| w as i32).collect();
    let arrays: [(&str, Vec<i32>); 3] = [
        ("exaone-moe.attention.sliding_window", windows),
        (
            "exaone-moe.attention.sliding_window_pattern",
            LAYERS.iter().map(|&(w, _, _)| i32::from(w > 0)).collect(),
        ),
        (
            "exaone-moe.attention.rope_pattern",
            LAYERS.iter().map(|&(_, r, _)| i32::from(r)).collect(),
        ),
    ];
    let clamps: Vec<f32> = LAYERS.iter().map(|&(_, _, c)| c).collect();
    for (key, want) in [
        ("exaone-moe.swiglu_clamp_exp", clamps.clone()),
        ("exaone-moe.swiglu_clamp_shexp", clamps),
    ] {
        let got = match kv.get(key) {
            Some(Value::Array(v)) => v
                .iter()
                .map(|x| match x {
                    Value::F32(f) => *f,
                    other => panic!("{key} holds {other:?}"),
                })
                .collect::<Vec<_>>(),
            other => {
                bad.push(format!("{key} is {other:?}"));
                continue;
            }
        };
        for (l, (a, b)) in got.iter().zip(&want).enumerate() {
            if a.to_bits() != b.to_bits() {
                bad.push(format!("{key} is {a} at layer {l}, wanted {b}"));
            }
        }
    }
    for (key, want) in arrays {
        let Value::Array(got) = kv.get(key).expect(key) else {
            bad.push(format!("{key} is not an array"));
            continue;
        };
        let as_i32 = |v: &Value| match v {
            Value::U32(x) => *x as i32,
            Value::Bool(b) => i32::from(*b),
            other => panic!("{key} holds {other:?}"),
        };
        let got: Vec<i32> = got.iter().map(as_i32).collect();
        if got != want {
            bad.push(format!("{key} is {got:?}, wanted {want:?}"));
        }
    }
    if let Some(Value::String(t)) = kv.get("tokenizer.chat_template")
        && t.len() != TEMPLATE
    {
        bad.push(format!("the chat template is {} bytes", t.len()));
    }

    // The description through the engine's own reader.
    let split = header_of(|_| {});
    let read = model::arch::spec(&split).expect("the exaone-moe reader");
    let s = &read.spec;
    if s.arch != models::Arch::ExaoneMoe {
        bad.push(format!("arch {:?}", s.arch));
    }
    if (s.hidden, s.vocab, s.ctx_train) != (6144, 153_600, 262_144) {
        bad.push(format!(
            "hidden {} vocab {} ctx {}",
            s.hidden, s.vocab, s.ctx_train
        ));
    }
    if s.rms_eps != 1e-5 || s.layers.len() != TRUNK || s.mtp.len() != 1 {
        bad.push(format!(
            "eps {} {} trunk layers, {} mtp",
            s.rms_eps,
            s.layers.len(),
            s.mtp.len()
        ));
    }
    if s.hc.is_some() || s.engram.is_some() || s.encoder.is_some() {
        bad.push(format!(
            "hc {:?} engram {:?} encoder {:?}",
            s.hc, s.engram, s.encoder
        ));
    }
    let template = s.chat.template.as_deref().unwrap_or("");
    if s.chat.pre != "exaone-moe"
        || s.chat.tools != Some(models::ToolFormat::QwenXml)
        || s.chat.reasoning != Some(models::ReasoningFormat::THINK)
        || template.len() != TEMPLATE
        || !template.contains("<tool_call>")
        || !template.contains("<function=")
        || !template.contains("<parameter=")
        || !template.contains("<think>")
        || !template.contains("</think>")
    {
        bad.push(format!(
            "chat pre {} tools {:?} reasoning {:?}, a template of {} bytes",
            s.chat.pre,
            s.chat.tools,
            s.chat.reasoning,
            template.len()
        ));
    }
    for (l, layer) in s.layers.iter().chain(&s.mtp).enumerate() {
        let (window, rope, clamp) = LAYERS[l];
        let Mixer::Gqa(g) = &layer.mixer else {
            bad.push(format!("layer {l} not GQA"));
            continue;
        };
        let want_window = (window > 0).then_some(window);
        let want = (
            g.heads,
            g.kv_heads,
            g.head_dim,
            g.value_dim,
            g.window,
            g.rope.mode,
            g.rope.dims,
        );
        let got = if rope {
            (64, 8, 128, 128, want_window, RopeMode::Neox, 128)
        } else {
            (64, 8, 128, 128, want_window, RopeMode::None, 0)
        };
        if want != got || g.rope.base != (if rope { 1e6 } else { 0.0 }) {
            bad.push(format!("layer {l} attention {g:?}"));
        }
        if !g.qk_norm || g.sinks {
            bad.push(format!(
                "layer {l}: qk_norm {} sinks {}",
                g.qk_norm, g.sinks
            ));
        }
        let dense = !(2..TRUNK).contains(&l);
        let act = if clamp > 0.0 {
            Act::SwiGluGateFirst { limit: clamp }
        } else {
            Act::SwiGlu { limit: None }
        };
        match &layer.ffn {
            Ffn::Dense { ff, act: a } if dense => {
                if *ff != 1024 || *a != (Act::SwiGlu { limit: None }) {
                    bad.push(format!("layer {l} dense {ff:?} {a:?}"));
                }
            }
            Ffn::Moe(m) if !dense => {
                if (m.experts, m.top_k, m.expert_ff) != (256, 8, 512) || m.act != act {
                    bad.push(format!("layer {l} moe {m:?}"));
                }
                if m.router
                    != (models::Router {
                        score: models::Score::Sigmoid,
                        bias: true,
                        norm: RouterNorm::SelectedSum,
                        scale: 2.5,
                        hash: false,
                    })
                {
                    bad.push(format!("layer {l} router {:?}", m.router));
                }
                let shared = m.shared.as_ref().unwrap();
                if (shared.ff, shared.act) != (512, act) || shared.sigmoid_gate {
                    bad.push(format!("layer {l} shared {shared:?}"));
                }
            }
            other => bad.push(format!("layer {l} ffn {other:?}")),
        }
        if layer.residual != models::Residual::Plain || !layer.extras.is_empty() {
            bad.push(format!("layer {l} residual {:?}", layer.residual));
        }
    }
    let needs = models::needs(s);
    if !needs.iter().any(|(n, _)| {
        matches!(
            n,
            Need::Router(m) if m.experts == 256 && m.top_k == 8
        )
    }) {
        bad.push("the description states no 256/top-8 router need".into());
    }
    if needs
        .iter()
        .filter(|(n, _)| {
            matches!(
                n,
                Need::ExpertAct {
                    act: Act::SwiGluGateFirst { .. },
                    ..
                }
            )
        })
        .count()
        != 4
    {
        bad.push("the clamped layers state no four gate-first expert acts".into());
    }

    // The plan's own values.
    let inputs = model::arch::flint::place::PlanInputs::read(&split)
        .expect("the composed program accepts the fixture");
    if program_of(s.arch) != Some(model::placement::accepts::Program::FlintBody) {
        bad.push("the architecture's program is not the flint body".into());
    }
    if inputs.hp
        != (model::arch::flint::place::MoeWidths {
            experts: 256,
            used: 8,
            expert_ff: 512,
            shared_ff: 512,
        })
    {
        bad.push(format!("routed widths {:?}", inputs.hp));
    }
    if model::arch::flint::place::rope_of(s).unwrap() != (1e6, 128) {
        bad.push("the trunk's one rope is not (1e6, 128)".into());
    }
    let row = model::arch::flint::place::router_row_of(s).unwrap();
    if (row.body, row.experts(), row.top_k_min, row.top_k_max)
        != (models::shape::RouterBody::BiasedSigmoid, 256, 8, 8)
    {
        bad.push(format!("the router instance {row:?}"));
    }
    for (l, want) in ROWS_AT_8192.iter().enumerate() {
        let got = inputs.kv.rows_at(l, 8192);
        if got != *want {
            bad.push(format!("layer {l} holds {got} rows at 8192, wanted {want}"));
        }
    }

    // The tensors: name, role, layer, type and shape, by the stems each kind
    // of block carries.
    let t = &read.tensors;
    if t.tensors.len() != 104 {
        bad.push(format!("{} classified tensors", t.tensors.len()));
    }
    let tensor = |name: &str| {
        t.tensors
            .iter()
            .find(|x| x.name == name)
            .map(|x| (x.role, x.layer, x.ty, x.dims.clone()))
    };
    for (name, want) in [
        (
            "token_embd.weight",
            (
                models::Role::TokenEmbedding,
                None,
                GgmlType::Q8_0,
                vec![6144, 153_600],
            ),
        ),
        (
            "output.weight",
            (
                models::Role::Head,
                None,
                GgmlType::Q8_0,
                vec![6144, 153_600],
            ),
        ),
        (
            "output_norm.weight",
            (models::Role::Head, None, GgmlType::F32, vec![6144]),
        ),
        (
            "blk.0.attn_q.weight",
            (
                models::Role::Attention,
                Some(0),
                GgmlType::Q8_0,
                vec![6144, 8192],
            ),
        ),
        (
            "blk.0.attn_k_norm.weight",
            (models::Role::Attention, Some(0), GgmlType::F32, vec![128]),
        ),
        (
            "blk.0.ffn_gate.weight",
            (
                models::Role::DenseFfn,
                Some(0),
                GgmlType::Q8_0,
                vec![6144, 1024],
            ),
        ),
        (
            "blk.2.ffn_gate_inp.weight",
            (
                models::Role::Router,
                Some(2),
                GgmlType::F32,
                vec![6144, 256],
            ),
        ),
        (
            "blk.2.exp_probs_b.bias",
            (models::Role::Router, Some(2), GgmlType::F32, vec![256]),
        ),
        (
            "blk.2.ffn_gate_exps.weight",
            (
                models::Role::RoutedExperts,
                Some(2),
                GgmlType::Q4_K,
                vec![6144, 512, 256],
            ),
        ),
        (
            "blk.2.ffn_down_exps.weight",
            (
                models::Role::RoutedExperts,
                Some(2),
                GgmlType::Q4_K,
                vec![512, 6144, 256],
            ),
        ),
        (
            "blk.2.ffn_gate_shexp.weight",
            (
                models::Role::SharedExpert,
                Some(2),
                GgmlType::Q8_0,
                vec![6144, 512],
            ),
        ),
        (
            "blk.5.ffn_norm.weight",
            (models::Role::FfnNorm, Some(5), GgmlType::F32, vec![6144]),
        ),
        (
            "blk.6.nextn.eh_proj.weight",
            (
                models::Role::Unused,
                Some(6),
                GgmlType::Q8_0,
                vec![12_288, 6144],
            ),
        ),
        (
            "blk.6.nextn.shared_head_norm.weight",
            (models::Role::Unused, Some(6), GgmlType::F32, vec![6144]),
        ),
        (
            "blk.6.ffn_up.weight",
            (
                models::Role::Unused,
                Some(6),
                GgmlType::Q8_0,
                vec![6144, 1024],
            ),
        ),
    ] {
        if tensor(name) != Some(want.clone()) {
            bad.push(format!("{name} {:?}, wanted {want:?}", tensor(name)));
        }
    }
    // Every tensor's type by its stem, every block's stem count by its kind.
    fn stems_of(name: &str) -> &str {
        name.strip_prefix("blk.")
            .and_then(|rest| rest.split_once('.'))
            .map_or(name, |(_, stem)| stem)
    }
    for x in &t.tensors {
        let stem = stems_of(&x.name);
        let want = match stem {
            "token_embd.weight" | "output.weight" => GgmlType::Q8_0,
            s if s.ends_with("_exps.weight") => GgmlType::Q4_K,
            "attn_q.weight"
            | "attn_k.weight"
            | "attn_v.weight"
            | "attn_output.weight"
            | "ffn_gate.weight"
            | "ffn_up.weight"
            | "ffn_down.weight"
            | "ffn_gate_shexp.weight"
            | "ffn_up_shexp.weight"
            | "ffn_down_shexp.weight"
            | "nextn.eh_proj.weight" => GgmlType::Q8_0,
            _ => GgmlType::F32,
        };
        if x.ty != want {
            bad.push(format!("{} is {:?}, the mix's {}", x.name, x.ty, want));
        }
    }
    for (l, want) in [
        (0, 11),
        (1, 11),
        (2, 16),
        (3, 16),
        (4, 16),
        (5, 16),
        (6, 15),
    ] {
        let got = t.tensors.iter().filter(|x| x.layer == Some(l)).count();
        if got != want {
            bad.push(format!("block {l} holds {got} tensors, wanted {want}"));
        }
    }

    // The type admission: a tensor in a type the architecture's stem table
    // does not hold its stem in is refused at the read, by name, with the
    // types the stem takes. The quantizer's first rules wrote the MTP input
    // projection Q4_K and nothing refused the file.
    for (name, ty, wants) in [
        (
            "blk.6.nextn.eh_proj.weight",
            GgmlType::Q4_K,
            "q4_K (the exaone-moe reader runs nextn.eh_proj.weight as q8_0 or bf16)",
        ),
        (
            "blk.2.ffn_gate_exps.weight",
            GgmlType::Q8_0,
            "q8_0 (the exaone-moe reader runs ffn_gate_exps.weight as q4_K or iq2_xxs or iq1_s or fp8_b128)",
        ),
    ] {
        let split = header_of(|decls| retype(decls, name, ty));
        match model::arch::spec(&split) {
            Err(e) => {
                let e = e.to_string();
                if !e.contains(name) || !e.contains(wants) {
                    bad.push(format!(
                        "{name} as {ty:?}: refused as {e}, wanted {wants:?}"
                    ));
                }
            }
            Ok(_) => bad.push(format!("{name} as {ty:?} was not refused")),
        }
    }

    // A file of another architecture is refused: the registry hands it to that
    // architecture's own reader (the flint one), whose keys this file does not
    // carry.
    let mut meta = gguf::inventory_of(&path).unwrap().meta;
    for (k, v) in &mut meta {
        if k == GENERAL_ARCHITECTURE {
            *v = Value::String("flint".into());
        }
    }
    let decls: Vec<TensorDecl> = gguf::inventory_of(&path)
        .unwrap()
        .tensors
        .iter()
        .map(|x| TensorDecl {
            name: x.name.clone(),
            dims: x.dims.clone(),
            type_id: x.type_id,
            nbytes: x.nbytes.unwrap(),
        })
        .collect();
    let layout = Layout::new(&meta, decls).unwrap();
    let split = Split::header_only(
        std::env::temp_dir().join(format!("bloomery-exaone-not-{}.gguf", std::process::id())),
        &layout,
    )
    .unwrap();
    match model::arch::spec(&split) {
        Err(e) => {
            let e = e.to_string();
            if !e.contains("flint.block_count") {
                bad.push(format!("the relabelled file: refused as {e}"));
            }
        }
        Ok(_) => bad.push("the relabelled file read as exaone-moe".into()),
    }

    println!(
        "exaone-moe fixture: header {HEADER_BYTES} B sha {HEADER_SHA:.12}… kv {} tensors {} \
         layers {TRUNK}+1 template {TEMPLATE} B, pins_differ {}",
        inv.meta.len(),
        inv.tensors.len(),
        bad.len()
    );
    assert!(
        bad.is_empty(),
        "{} pin(s) differ:\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}

/// The program a description of `arch` runs, without the placement's own
/// error type.
fn program_of(arch: models::Arch) -> Option<model::placement::accepts::Program> {
    model::placement::accepts::program_of(arch)
}
