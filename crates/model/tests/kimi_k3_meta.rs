//! kimi_k3 metadata gate: what the header reader resolves from the Kimi K3 UD-IQ1_S file
//! (unsloth/Kimi-K3-GGUF), whose header tables are checked in beside this test
//! (`kimi_k3_hdr.kv.json`: the keys of the first shard, which holds no tensor; `kimi_k3_tensors.tsv`:
//! every tensor of shards 2 to 14, the 2,573 of the file). The test builds the header alone as a
//! GGUF (`common/header_tables.rs`: shapes, types and keys, no tensor's bytes) and reads it through
//! the engine's own `Split`; the reader holds every tensor to the dims the keys give it and to the
//! quant types the released set holds its stem in, so the headers are the whole file as far as it
//! reaches. One contract, one test: each prints its summary, then fails with every pin that differs.
//!
//! No program runs the architecture yet, so the coverage table's list is not read here.

use gguf::{GENERAL_ARCHITECTURE, GgmlType, Split, Value};
use model::arch::kimi_k3::fixture as k3;
use model::fixture::{FilePlan, FixtureSpec};
use model::placement::Role;
use models::{Act, DeltaKind, Ffn, KdaGate, LatentUp, Mixer, Moe, OutGate, Residual, Router};
use models::{RouterNorm, Score, Shared};

#[path = "common/header_tables.rs"]
mod header_tables;

/// The checked-in header tables.
const KV: &str = include_str!("kimi_k3_hdr.kv.json");
const TENSORS: &str = include_str!("kimi_k3_tensors.tsv");

/// The layers 0-based that mix with the latent attention (`attention.head_count_kv` 1): every
/// fourth from 3 through 91, and the last, 92.
fn latent(l: usize) -> bool {
    l == 92 || (l % 4 == 3 && l <= 91)
}

/// The layers whose routed gate and up stacks are IQ2_XXS and down stack IQ1_S, by the quant's
/// own recipe.
const MIX_GATE_UP_XXS: &[usize] = &[11, 46, 47, 56, 57, 58, 68, 69, 70, 71];

/// The layers whose routed gate and up stacks are IQ1_S and down stack IQ2_XXS.
const MIX_DOWN_XXS: &[usize] = &[83, 92];

/// The shard's header as a `Split` holding no tensor's bytes.
fn header() -> Split {
    header_tables::header_of("k3-meta", KV, TENSORS)
}

#[test]
fn kimi_k3_header() {
    let split = header();
    let read = model::arch::spec(&split).unwrap_or_else(|e| panic!("spec: {e}"));
    let hp = model::arch::kimi_k3::hparams::Hparams::read(&split)
        .unwrap_or_else(|e| panic!("hparams: {e}"));
    let mut unread = model::arch::kimi_k3::hparams::unread_keys(&split);
    unread.sort();
    let s = &read.spec;
    let mut bad = Vec::new();
    println!(
        "kimi-k3 kvs={} tensors={} layers={} latent={} banks={} hidden={} vocab={} ctx={} \
         template={} unread={unread:?}",
        split.shard(0).expect("the one shard").kv_count(),
        split.tensor_count(),
        s.layers.len(),
        (0..s.layers.len()).filter(|&l| latent(l)).count(),
        (0..s.layers.len()).filter(|&l| hp.banks(l)).count(),
        s.hidden,
        s.vocab,
        s.ctx_train,
        s.chat.template.as_ref().map_or(0, |t| t.len()),
    );

    if s.arch != models::Arch::KimiK3 {
        bad.push(format!("arch {:?}", s.arch));
    }
    if (s.hidden, s.vocab, s.ctx_train) != (7168, 163_840, 1_048_576) {
        bad.push(format!(
            "hidden {} vocab {} ctx {}",
            s.hidden, s.vocab, s.ctx_train
        ));
    }
    if s.rms_eps.to_bits() != hp.rms_eps.to_bits() || hp.rms_eps != 1e-5 {
        bad.push(format!("rms_eps {} (hp {:?})", s.rms_eps, hp.rms_eps));
    }
    if !read.defaults.is_empty() {
        bad.push(format!("defaults {:?}", read.defaults));
    }
    // The file's `attention.value_length` (74) does not describe the cache and its `rope.freq_base`
    // no layer reads: listed, never refused.
    if unread != ["kimi-k3.attention.value_length", "kimi-k3.rope.freq_base"] {
        bad.push(format!("unread keys {unread:?}"));
    }
    if s.chat.pre != "kimi-k2"
        || s.chat.tools.is_some()
        || s.chat.reasoning.is_some()
        || s.chat.template.as_ref().is_none_or(|t| t.len() != 24_696)
    {
        bad.push(format!(
            "chat pre {} tools {:?} reasoning {:?} template {:?}",
            s.chat.pre,
            s.chat.tools,
            s.chat.reasoning,
            s.chat.template.as_ref().map(|t| t.len())
        ));
    }
    if s.layers.len() != 93
        || !s.mtp.is_empty()
        || s.hc.is_some()
        || s.engram.is_some()
        || s.encoder.is_some()
        || s.logit_epilogue.is_some()
    {
        bad.push(format!(
            "{} layers, mtp {} hc {:?} engram {:?}",
            s.layers.len(),
            s.mtp.len(),
            s.hc,
            s.engram
        ));
    }
    let act = Act::Situ {
        gate_beta: 4.0,
        value_beta: 25.0,
    };
    let want_moe = Moe {
        experts: 896,
        top_k: 16,
        expert_ff: 3072,
        latent: Some(3584),
        pre_exp_norm: false,
        post_exp_norm: None,
        sum_norm: true,
        act,
        router: Router {
            score: Score::Sigmoid,
            bias: true,
            norm: RouterNorm::SelectedSum,
            scale: 1.0,
            hash: false,
        },
        shared: Some(Shared {
            ff: 6144,
            act,
            sigmoid_gate: false,
        }),
    };
    for (l, layer) in s.layers.iter().enumerate() {
        if layer.residual != (Residual::AttnRes { block: 12 }) || !layer.extras.is_empty() {
            bad.push(format!("layer {l} residual {:?}", layer.residual));
        }
        match (&layer.mixer, latent(l)) {
            (Mixer::DeltaRule(d), false) => {
                if d.kind
                    != (DeltaKind::KdaFullGate {
                        gate: KdaGate::Sigmoid { lower_bound: -5.0 },
                    })
                    || (d.k_heads, d.v_heads, d.d, d.conv) != (96, 96, 128, 4)
                {
                    bad.push(format!("layer {l} delta rule {d:?}"));
                }
            }
            (Mixer::Latent(a), true) => {
                if (a.heads, a.q_lora, a.latent, a.pe) != (96, 1536, 512, 64)
                    || a.up != (LatentUp::Absorbed { qk: 192, v: 128 })
                    || a.out != models::LatentOut::Plain
                    || a.out_gate != OutGate::Proj
                    || a.rope.is_some()
                    || a.q_head_norm
                    || a.ssmax
                    || a.window.is_some()
                    || a.sinks
                    || a.compress.is_some()
                    || a.select.is_some()
                    || a.cache != models::Source::Own(())
                {
                    bad.push(format!("layer {l} latent {a:?}"));
                }
            }
            (other, _) => bad.push(format!("layer {l}: {}", other.what())),
        }
        if l == 0 {
            if layer.ffn != (Ffn::Dense { ff: 33_792, act }) {
                bad.push(format!("layer 0 ffn {:?}", layer.ffn));
            }
            continue;
        }
        match &layer.ffn {
            Ffn::Moe(m) if *m == want_moe => {}
            other => bad.push(format!("layer {l} ffn {other:?}")),
        }
    }
    let banks: Vec<usize> = (0..93).filter(|&l| hp.banks(l)).collect();
    if banks != [0, 12, 24, 36, 48, 60, 72, 84] {
        bad.push(format!("banks {banks:?}"));
    }
    if hp.kda_state() != [96, 128, 128] {
        bad.push(format!("kda state {:?}", hp.kda_state()));
    }

    // The tensors, by the stems each kind of layer carries: 29 a routed delta layer, 24 a routed
    // latent one, 21 the dense lead, and the four model-level ones.
    let t = &read.tensors;
    if t.tensors.len() != 2573 {
        bad.push(format!("{} tensors, the file holds 2573", t.tensors.len()));
    }
    let want_count: usize = 4
        + (0..93)
            .map(|l| match (latent(l), l == 0) {
                (false, true) => 21,
                (false, false) => 29,
                (true, _) => 24,
            })
            .sum::<usize>();
    if want_count != 2573 {
        bad.push(format!("the stems of the 93 layers add to {want_count}"));
    }
    let count = |role: Role| t.tensors.iter().filter(|x| x.role == role).count();
    for (role, want) in [
        (Role::RoutedExperts, 276),
        (Role::SharedExpert, 276),
        (Role::DenseFfn, 3),
    ] {
        if count(role) != want {
            bad.push(format!("{role:?}: {} tensors, wanted {want}", count(role)));
        }
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
                Role::TokenEmbedding,
                None,
                GgmlType::Q6_K,
                vec![7168, 163_840],
            ),
        ),
        (
            "output_res_score.weight",
            (Role::Head, None, GgmlType::F32, vec![7168]),
        ),
        (
            "blk.0.ssm_g.weight",
            (Role::Attention, Some(0), GgmlType::Q6_K, vec![7168, 12_288]),
        ),
        (
            "blk.0.ssm_f_b.weight",
            (Role::Attention, Some(0), GgmlType::Q8_0, vec![128, 12_288]),
        ),
        (
            "blk.0.attn_res_score.weight",
            (Role::Attention, Some(0), GgmlType::F32, vec![7168]),
        ),
        (
            "blk.0.ffn_res_score.weight",
            (Role::FfnNorm, Some(0), GgmlType::F32, vec![7168]),
        ),
        (
            "blk.0.ffn_down.weight",
            (Role::DenseFfn, Some(0), GgmlType::Q6_K, vec![33_792, 7168]),
        ),
        (
            "blk.1.ffn_gate_exps.weight",
            (
                Role::RoutedExperts,
                Some(1),
                GgmlType::IQ1_S,
                vec![3584, 3072, 896],
            ),
        ),
        (
            "blk.1.ffn_down_exps.weight",
            (
                Role::RoutedExperts,
                Some(1),
                GgmlType::IQ1_S,
                vec![3072, 3584, 896],
            ),
        ),
        (
            "blk.1.ffn_gate_inp.weight",
            (Role::Router, Some(1), GgmlType::F32, vec![7168, 896]),
        ),
        (
            "blk.1.ffn_routed_down.weight",
            (Role::Router, Some(1), GgmlType::Q6_K, vec![7168, 3584]),
        ),
        (
            "blk.1.ffn_routed_up.weight",
            (Role::Router, Some(1), GgmlType::Q6_K, vec![3584, 7168]),
        ),
        (
            "blk.1.ffn_routed_norm.weight",
            (Role::Router, Some(1), GgmlType::F32, vec![3584]),
        ),
        (
            "blk.1.ffn_gate_shexp.weight",
            (
                Role::SharedExpert,
                Some(1),
                GgmlType::Q6_K,
                vec![7168, 6144],
            ),
        ),
        (
            "blk.3.attn_k_b.weight",
            (Role::Attention, Some(3), GgmlType::Q8_0, vec![128, 512, 96]),
        ),
        (
            "blk.3.attn_v_b.weight",
            (Role::Attention, Some(3), GgmlType::Q6_K, vec![512, 128, 96]),
        ),
        (
            "blk.3.attn_q_b.weight",
            (Role::Attention, Some(3), GgmlType::Q6_K, vec![1536, 18_432]),
        ),
        (
            "blk.3.attn_kv_a_mqa.weight",
            (Role::Attention, Some(3), GgmlType::Q6_K, vec![7168, 576]),
        ),
        (
            "blk.3.attn_gate.weight",
            (Role::Attention, Some(3), GgmlType::Q6_K, vec![7168, 12_288]),
        ),
        (
            "blk.1.ssm_conv1d_q.weight",
            (Role::Attention, Some(1), GgmlType::F32, vec![4, 1, 12_288]),
        ),
        (
            "blk.46.ffn_gate_exps.weight",
            (
                Role::RoutedExperts,
                Some(46),
                GgmlType::IQ2_XXS,
                vec![3584, 3072, 896],
            ),
        ),
        (
            "blk.83.ffn_down_exps.weight",
            (
                Role::RoutedExperts,
                Some(83),
                GgmlType::IQ2_XXS,
                vec![3072, 3584, 896],
            ),
        ),
    ] {
        if tensor(name) != Some(want.clone()) {
            bad.push(format!("{name} {:?}, wanted {want:?}", tensor(name)));
        }
    }
    // The routed stacks' types by layer: the quant's own recipe, not one type for every layer.
    let stack_ty = |l: usize, stem: &str| tensor(&format!("blk.{l}.{stem}.weight")).map(|x| x.2);
    for l in 1..93 {
        let (gate_up, down) = match (MIX_GATE_UP_XXS.contains(&l), MIX_DOWN_XXS.contains(&l)) {
            (true, _) => (GgmlType::IQ2_XXS, GgmlType::IQ1_S),
            (_, true) => (GgmlType::IQ1_S, GgmlType::IQ2_XXS),
            _ => (GgmlType::IQ1_S, GgmlType::IQ1_S),
        };
        let got = (
            stack_ty(l, "ffn_gate_exps"),
            stack_ty(l, "ffn_up_exps"),
            stack_ty(l, "ffn_down_exps"),
        );
        if got != (Some(gate_up), Some(gate_up), Some(down)) {
            bad.push(format!("layer {l} stacks {got:?}"));
        }
    }
    // Every other layer kind holds its own stems and none of the other's.
    for (l, name) in [
        (0, "blk.0.attn_gate.weight"),
        (3, "blk.3.ssm_g.weight"),
        (3, "blk.3.attn_q.weight"),
        (0, "blk.0.ffn_gate_inp.weight"),
        (0, "blk.0.ffn_routed_down.weight"),
        (1, "blk.1.ffn_gate.weight"),
    ] {
        if tensor(name).is_some() {
            bad.push(format!("layer {l}: {name} is in the file"));
        }
    }

    assert!(
        bad.is_empty(),
        "{} pin(s) differ:\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}

/// The fixture's header as a `Split` holding no tensor's bytes: the plan's one shard laid out.
fn fixture_header(plan: &FilePlan) -> Split {
    let layouts = plan.layouts().expect("the plan lays out");
    assert_eq!(layouts.len(), 1, "the fixture is one shard");
    let path =
        std::env::temp_dir().join(format!("bloomery-k3-fixture-{}.gguf", std::process::id()));
    Split::header_only(path, &layouts[0].1).expect("the fixture's header-only split opens")
}

/// The fixture clause: the plan of the fixture of the real header's four layers `[0, 46, 83, 84]`
/// (`model::arch::kimi_k3::fixture`), laid out as a header-only file, reads through the engine's
/// own reader as the cut. The kinds are the real file's at the map's layers, derived here from the
/// pins above, not from the fixture's own table: delta, delta, latent, delta, the dense lead
/// first, a bank at the first and third; the mini's widths are its keys' and every tensor holds the
/// real layer's type. A map of other kinds or other stack types, a head-count array left at the
/// source's, a key left at the source's, a removed, reshaped or retyped tensor and a missing key
/// are each refused by name.
#[test]
fn kimi_k3_fixture_cut() {
    let source = header();
    let spec = k3::spec();
    let plan = model::fixture::plan(&spec, &source, None, &spec.options())
        .unwrap_or_else(|e| panic!("plan: {e}"));
    let fx = fixture_header(&plan.target);
    let hp = model::arch::kimi_k3::hparams::Hparams::read(&fx)
        .unwrap_or_else(|e| panic!("the fixture's hparams: {e}"));
    let read = model::arch::spec(&fx).unwrap_or_else(|e| panic!("the fixture's spec: {e}"));
    let s = &read.spec;
    let mut bad = Vec::new();

    // The kinds, from the pins of the real header.
    let want_kinds: Vec<_> = k3::LAYER_MAP
        .iter()
        .map(|&l| {
            if latent(l) {
                model::arch::kimi_k3::hparams::Kind::Mla
            } else {
                model::arch::kimi_k3::hparams::Kind::Kda
            }
        })
        .collect();
    if hp.kinds != want_kinds {
        bad.push(format!("kinds {:?}, the cut's {want_kinds:?}", hp.kinds));
    }
    {
        use model::arch::kimi_k3::hparams::Kind::{Kda, Mla};
        if want_kinds != [Kda, Kda, Mla, Kda] {
            bad.push(format!(
                "the cut's kinds {want_kinds:?} are not the four kinds"
            ));
        }
    }
    let banks: Vec<bool> = (0..4).map(|l| hp.banks(l)).collect();
    if banks != [true, false, true, false] {
        bad.push(format!("banks {banks:?}"));
    }

    // The mini's widths and the facts kept.
    let got = (
        hp.n_layer,
        hp.n_embd,
        hp.n_head,
        hp.experts.n_expert,
        hp.experts.n_used,
        hp.experts.ff,
        hp.experts.shared_ff(),
        hp.latent,
        hp.dense_ff,
        hp.mla.q_lora,
        hp.attn_res_block,
    );
    if got != (4, 512, 2, 8, 2, 256, 512, 256, 1024, 256, 2) {
        bad.push(format!("mini widths {got:?}"));
    }
    let kept = (
        hp.kda.head_dim,
        hp.kda.conv,
        hp.mla.kv_lora,
        hp.mla.head_k,
        hp.mla.head_v,
        hp.mla.pe,
        hp.n_vocab,
        hp.n_ctx_train,
        hp.dense_lead,
        hp.experts.n_shared,
    );
    if kept != (128, 4, 512, 192, 128, 64, 163_840, 1_048_576, 1, 2) {
        bad.push(format!("kept facts {kept:?}"));
    }
    if hp.gate != (KdaGate::Sigmoid { lower_bound: -5.0 }) || !hp.defaults.is_empty() {
        bad.push(format!("gate {:?} defaults {:?}", hp.gate, hp.defaults));
    }
    if (s.hidden, s.vocab) != (512, 163_840) || s.layers.len() != 4 {
        bad.push(format!("spec {} {} {}", s.hidden, s.vocab, s.layers.len()));
    }

    // Every tensor at the real layer's type, the stems of each kind.
    let t = &read.tensors;
    if t.tensors.len() != 107 {
        bad.push(format!(
            "{} tensors, the four layers hold 107",
            t.tensors.len()
        ));
    }
    let tensor = |name: &str| {
        t.tensors
            .iter()
            .find(|x| x.name == name)
            .map(|x| (x.ty, x.dims.clone()))
    };
    for (name, want) in [
        ("token_embd.weight", (GgmlType::Q6_K, vec![512, 163_840])),
        ("output.weight", (GgmlType::Q6_K, vec![512, 163_840])),
        ("output_res_score.weight", (GgmlType::F32, vec![512])),
        ("blk.0.ssm_g.weight", (GgmlType::Q6_K, vec![512, 256])),
        ("blk.0.ssm_f_b.weight", (GgmlType::Q8_0, vec![128, 256])),
        ("blk.0.ffn_down.weight", (GgmlType::Q6_K, vec![1024, 512])),
        (
            "blk.1.ssm_conv1d_q.weight",
            (GgmlType::F32, vec![4, 1, 256]),
        ),
        ("blk.1.ssm_a", (GgmlType::F32, vec![2])),
        (
            "blk.1.ffn_gate_exps.weight",
            (GgmlType::IQ2_XXS, vec![256, 256, 8]),
        ),
        (
            "blk.1.ffn_down_exps.weight",
            (GgmlType::IQ1_S, vec![256, 256, 8]),
        ),
        ("blk.1.ffn_gate_inp.weight", (GgmlType::F32, vec![512, 8])),
        (
            "blk.1.ffn_routed_up.weight",
            (GgmlType::Q6_K, vec![256, 512]),
        ),
        ("blk.2.attn_k_b.weight", (GgmlType::Q8_0, vec![128, 512, 2])),
        ("blk.2.attn_v_b.weight", (GgmlType::Q6_K, vec![512, 128, 2])),
        ("blk.2.attn_q_b.weight", (GgmlType::Q6_K, vec![256, 384])),
        (
            "blk.2.attn_kv_a_mqa.weight",
            (GgmlType::Q6_K, vec![512, 576]),
        ),
        (
            "blk.2.ffn_gate_exps.weight",
            (GgmlType::IQ1_S, vec![256, 256, 8]),
        ),
        (
            "blk.2.ffn_down_exps.weight",
            (GgmlType::IQ2_XXS, vec![256, 256, 8]),
        ),
        (
            "blk.3.ffn_up_exps.weight",
            (GgmlType::IQ1_S, vec![256, 256, 8]),
        ),
        (
            "blk.3.ffn_gate_shexp.weight",
            (GgmlType::Q6_K, vec![512, 512]),
        ),
        (
            "blk.3.ffn_down_shexp.weight",
            (GgmlType::Q6_K, vec![512, 512]),
        ),
    ] {
        if tensor(name) != Some(want.clone()) {
            bad.push(format!("{name} {:?}, wanted {want:?}", tensor(name)));
        }
    }
    for name in [
        "blk.2.ssm_g.weight",
        "blk.1.attn_gate.weight",
        "blk.0.ffn_gate_inp.weight",
        "blk.0.ffn_routed_down.weight",
    ] {
        if tensor(name).is_some() {
            bad.push(format!("{name} is in the fixture"));
        }
    }
    if let Err(e) = k3::check_kinds(&spec, &fx, &source) {
        bad.push(format!("check_kinds: {e}"));
    }

    // The refusals, each by name.
    let mutant = |what: &str, want: &str, edit: &dyn Fn(&mut FilePlan)| {
        let mut p = plan.target.clone();
        edit(&mut p);
        let split = fixture_header(&p);
        match k3::check_kinds(&spec, &split, &source) {
            Err(e) if e.to_string().contains(want) => Ok(()),
            Err(e) => Err(format!("{what}: refused as {e}, wanted {want:?}")),
            Ok(_) => Err(format!("{what}: accepted")),
        }
    };
    let set = |p: &mut FilePlan, key: &str, v: Value| {
        p.kvs
            .iter_mut()
            .find(|(k, _)| k == &format!("kimi-k3.{key}"))
            .unwrap_or_else(|| panic!("{key} is in the plan"))
            .1 = v;
    };
    let retype = |p: &mut FilePlan, name: &str, ty: GgmlType, dims: Option<Vec<u64>>| {
        let t = p
            .tensors
            .iter_mut()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("{name} is in the plan"));
        t.ty = ty;
        if let Some(dims) = dims {
            t.dims = dims;
        }
        let (_, blck, tsz) = gguf::ggml_type_info(t.ty.as_u32()).expect("a type");
        t.nbytes = gguf::tensor_bytes(blck, tsz, &t.dims).expect("bytes");
    };
    // The one shard's range follows the tensors the plan holds.
    let drop = |p: &mut FilePlan, name: &str| {
        p.tensors.retain(|t| t.name != name);
        p.shards.clear();
        p.shards.push(0..p.tensors.len());
    };
    let results = [
        mutant(
            "a delta layer's output gate removed",
            "blk.1.ssm_g.weight",
            &|p| drop(p, "blk.1.ssm_g.weight"),
        ),
        mutant(
            "a latent layer's output gate removed",
            "blk.2.attn_gate.weight",
            &|p| drop(p, "blk.2.attn_gate.weight"),
        ),
        mutant(
            "the final mix's scores removed",
            "output_res_score.weight",
            &|p| drop(p, "output_res_score.weight"),
        ),
        mutant("a tensor of another shape", "blk.1.attn_q.weight", &|p| {
            retype(
                p,
                "blk.1.attn_q.weight",
                GgmlType::Q6_K,
                Some(vec![512, 257]),
            );
        }),
        mutant(
            "a routed stack of another expert count",
            "blk.3.ffn_gate_exps.weight",
            &|p| {
                retype(
                    p,
                    "blk.3.ffn_gate_exps.weight",
                    GgmlType::IQ1_S,
                    Some(vec![256, 256, 9]),
                );
            },
        ),
        mutant(
            "the latent experts' key missing",
            "kimi-k3.expert_latent_length",
            &|p| p.kvs.retain(|(k, _)| k != "kimi-k3.expert_latent_length"),
        ),
        mutant(
            "the bank interval left at the source's",
            "attn_res.block_size",
            &|p| set(p, "attn_res.block_size", Value::U32(12)),
        ),
        mutant(
            "the head counts left at the source's 93",
            "attention.head_count_kv",
            &|p| {
                let kv: Vec<Value> = (0..93).map(|l| Value::I32(i32::from(latent(l)))).collect();
                set(p, "attention.head_count_kv", Value::Array(kv));
            },
        ),
        mutant("a retyped routed stack", "ffn_gate_exps.weight is", &|p| {
            retype(p, "blk.2.ffn_gate_exps.weight", GgmlType::IQ2_XXS, None);
        }),
        mutant(
            "a tensor of a type the released set does not hold",
            "released Kimi K3 set",
            &|p| retype(p, "blk.1.attn_q.weight", GgmlType::Q4_K, None),
        ),
    ];
    for r in results {
        if let Err(why) = r {
            bad.push(why);
        }
    }
    // A map of other kinds or stack types is refused before any byte is planned.
    for (layers, want) in [
        (
            vec![0, 1, 83, 84],
            "source layer 1 (fixture 1) holds ffn_gate_exps.weight as Some(IQ1_S)",
        ),
        (
            vec![0, 46, 82, 84],
            "source layer 82 (fixture 2) is a delta layer, the kind keeps it a latent layer",
        ),
        (
            vec![1, 46, 83, 84],
            "source layer 1 (fixture 0) is routed, the kind keeps it dense",
        ),
        (
            vec![0, 46, 83, 93],
            "source layer 93 is past the source's 93 layers",
        ),
    ] {
        let other = FixtureSpec {
            layers: layers.clone(),
            ..k3::spec()
        };
        match model::fixture::plan(&other, &source, None, &other.options()) {
            Err(e) if e.to_string().contains(want) => {}
            Err(e) => bad.push(format!(
                "the map {layers:?}: refused as {e}, wanted {want:?}"
            )),
            Ok(_) => bad.push(format!("the map {layers:?}: planned")),
        }
    }

    println!(
        "kimi-k3 fixture layers=4 tensors={} shards={} pins_differ={}",
        t.tensors.len(),
        plan.target.shards.len(),
        bad.len()
    );
    assert!(
        bad.is_empty(),
        "{} pin(s) differ:\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}

/// A file of another architecture is refused by `general.architecture`, whatever its keys and
/// tensors hold: the real tables relabelled as a V4.1 file and as a GLM file are not Kimi K3's, and
/// the reader names the string it found.
#[test]
fn another_architecture_is_not_kimi_k3() {
    for arch in ["deepseek41", "glm5next"] {
        let mut table = serde_json::from_str::<serde_json::Value>(KV).expect("the table parses");
        table[GENERAL_ARCHITECTURE] = serde_json::Value::String(arch.to_string());
        let split =
            header_tables::header_of(&format!("k3-not-{arch}"), &table.to_string(), TENSORS);
        match model::arch::kimi_k3::spec::read(&split) {
            Ok(_) => panic!("{arch}: read as Kimi K3"),
            Err(e) => {
                let e = e.to_string();
                assert!(
                    e.contains(GENERAL_ARCHITECTURE) && e.contains(arch),
                    "{arch}: {e}"
                );
            }
        }
    }
}
