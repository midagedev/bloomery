//! mistrallarge4 metadata gate: what the header reader resolves from the
//! Mistral Large 4 (1T-A52B) UD-IQ1_S file's first shard
//! (unsloth/Mistral-Large-4-1T-A52B-GGUF), whose header tables are checked in
//! beside this test (`mistrallarge4_hdr.kv.json`, `mistrallarge4_tensors.tsv`
//! — every key and tensor of the shard's header, the four tokenizer arrays
//! the fetch summarized away and nothing else left out). The test builds the
//! header alone as a GGUF ([`gguf::write::Layout`] +
//! [`gguf::Split::header_only`]: shapes, types and keys, no tensor's bytes)
//! and reads it through the engine's own `Split`; the reader holds every
//! tensor to the dims the keys give it, so the headers are the whole file as
//! far as it reaches. One contract, one test: it prints the summary, then
//! fails with every pin that differs.
//!
//! `Program::MistralLarge4Body` is the plan's program and no row of the
//! coverage table lists it yet, so the coverage check's list is printed and
//! pinned, not empty — as LFM3's gate holds its own. The context bound
//! (`spec::serve_ctx`) is pinned here too, and so is the plan of the header
//! on the gate plan and on plan (b′) (`mistrallarge4_plan_*`), for this file
//! and the Mix-256G file.

use gguf::write::{Layout, TensorDecl};
use gguf::{GgmlType, Split, Value};
use model::arch::mistrallarge4::fixture as ml4;
use model::fixture::{FilePlan, FixtureSpec};
use models::{
    Act, Ffn, IndexKeys, LatentUp, LogitEpilogue, Moe, OutGate, RouterNorm, Score, Selector, Source,
};

/// The checked-in header tables.
const KV: &str = include_str!("mistrallarge4_hdr.kv.json");
const TENSORS: &str = include_str!("mistrallarge4_tensors.tsv");

/// The NoPE layers (`attention.rope_pattern` false): every fourth from 3.
const NOPE: &[usize] = &[
    3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71,
];

/// The layers that read another layer's cache (`attention.kv_source_map`):
/// every fourth from 39, each of layer 35's.
const KV_READERS: &[usize] = &[39, 43, 47, 51, 55, 59, 63, 67, 71];

/// The layers that run their own indexer (`attention.indexer.types` true):
/// every one but the shared pairs from 17.
fn full(l: usize) -> bool {
    !((17..=70).contains(&l) && matches!(l % 4, 1 | 2))
}

/// The last full indexer layer below `l`.
fn prev_full(l: usize) -> usize {
    (0..l).rev().find(|&i| full(i)).expect("layer 0 is full")
}

/// The shard's header as a `Split` holding no tensor's bytes.
fn header() -> Split {
    header_of("meta", KV, TENSORS)
}

/// One line of a tensor table, name, type, dims and shard, as the declaration the header lays out
/// (its bytes the file's: whole blocks of every row) and its type.
fn decl(line: &str) -> (TensorDecl, GgmlType) {
    let (name, ty, dims, _) = line
        .split_once('\t')
        .and_then(|(name, rest)| {
            let (ty, rest) = rest.split_once('\t')?;
            let (dims, _) = rest.split_once('\t')?;
            Some((name, ty, dims, ()))
        })
        .unwrap_or_else(|| panic!("{line}: not name, type, dims, shard"));
    let dims: Vec<u64> = dims.split('x').map(|d| d.parse().expect("a dim")).collect();
    let ty = ty_of(ty);
    let rows = dims[1..].iter().product::<u64>();
    let per_row = ty.type_size().expect("a size") * (dims[0] / ty.blck_size().expect("a block"));
    let decl = TensorDecl {
        name: name.to_string(),
        dims,
        type_id: ty.as_u32(),
        nbytes: per_row * rows,
    };
    (decl, ty)
}

/// The header of the tables `kv` and `tensors` as a `Split` holding no tensor's bytes, laid out at
/// a path of its own under `tag` (the tests run in threads of one process). A key table with no
/// chat template (the Mix-256G dump cut it) takes the UD-IQ1_S file's: no plan reads its text.
fn header_of(tag: &str, kv: &str, tensors: &str) -> Split {
    let mut table =
        serde_json::from_str::<serde_json::Value>(kv).expect("the checked-in header table parses");
    let table = table
        .as_object_mut()
        .expect("the header table is an object");
    if !table.contains_key("tokenizer.chat_template") {
        let ud = serde_json::from_str::<serde_json::Value>(KV).expect("the UD table parses");
        let template = ud["tokenizer.chat_template"].clone();
        table.insert("tokenizer.chat_template".to_string(), template);
    }
    let kvs: Vec<(String, Value)> = table
        .iter()
        // The fetch summarized the four tokenizer arrays the reader never
        // reads (`{array_len, head}`); the token list's own key the synthetic
        // readers write is the reader's too, never read here.
        .filter_map(|(k, v)| value(k, v).map(|v| (k.clone(), v)))
        .collect();
    let tensors: Vec<TensorDecl> = tensors.lines().map(|line| decl(line).0).collect();
    let layout = Layout::new(&kvs, tensors).expect("the header lays out");
    let path = std::env::temp_dir().join(format!("bloomery-ml4-{tag}-{}.gguf", std::process::id()));
    Split::header_only(path, &layout).expect("the header-only split opens")
}

/// A header value as the GGUF's: ints as U32 or U64, floats as F32, strings,
/// bools and arrays of them; `None` for the summarized tokenizer tables.
fn value(key: &str, v: &serde_json::Value) -> Option<Value> {
    use serde_json::Value as J;
    match v {
        J::Bool(b) => Some(Value::Bool(*b)),
        J::String(s) => Some(Value::String(s.clone())),
        J::Number(n) => {
            if let Some(i) = n.as_u64() {
                Some(if i <= u64::from(u32::MAX) {
                    Value::U32(i as u32)
                } else {
                    Value::U64(i)
                })
            } else if let Some(i) = n.as_i64() {
                u64::try_from(i).ok().map(Value::U64)
            } else {
                n.as_f64().map(|f| Value::F32(f as f32))
            }
        }
        J::Array(items) => Some(Value::Array(
            items
                .iter()
                .map(|item| value(key, item))
                .collect::<Option<_>>()?,
        )),
        J::Object(_) => None,
        J::Null => panic!("{key}: the header table holds a null"),
    }
}

/// A ggml type by its table's name (the table spells the ids in capitals,
/// `ggml_type_name` in mixed case).
fn ty_of(name: &str) -> GgmlType {
    (0..=45)
        .chain(2000..=2005)
        .map(GgmlType::from_u32)
        .find(|ty| ty.name().is_some_and(|n| n.eq_ignore_ascii_case(name)))
        .unwrap_or_else(|| panic!("{name}: no ggml type of this name"))
}

#[test]
fn mistrallarge4_header() {
    let split = header();
    let read = model::arch::spec(&split).unwrap_or_else(|e| panic!("spec: {e}"));
    let hp = model::arch::mistrallarge4::hparams::Hparams::read(&split)
        .unwrap_or_else(|e| panic!("hparams: {e}"));
    let unread = model::arch::mistrallarge4::hparams::unread_keys(&split);
    let s = &read.spec;
    let items: Vec<String> = model::arch::coverage::check(s, &read.tensors)
        .into_iter()
        .map(|u| u.feature)
        .collect();
    let mut bad = Vec::new();
    println!(
        "mistral-large4 kvs={} tensors={} layers={} nope={} kv_readers={} full_indexers={} \
         hidden={} vocab={} ctx={} template={} tensors_table={} unread={unread:?}",
        split.shard(0).expect("the one shard").kv_count(),
        split.tensor_count(),
        s.layers.len(),
        NOPE.len(),
        KV_READERS.len(),
        (0..s.layers.len()).filter(|&l| full(l)).count(),
        s.hidden,
        s.vocab,
        s.ctx_train,
        s.chat.template.as_ref().map_or(0, |t| t.len()),
        read.tensors.tensors.len(),
    );
    let mut distinct: Vec<&String> = items.iter().collect();
    distinct.sort();
    distinct.dedup();
    let shown: Vec<&str> = distinct.iter().map(|f| f.as_str()).collect();
    println!(
        "coverage: {} items, distinct {:?} …",
        items.len(),
        &shown[..shown.len().min(30)]
    );

    if s.arch != models::Arch::MistralLarge4 {
        bad.push(format!("arch {:?}", s.arch));
    }
    if (s.hidden, s.vocab, s.ctx_train) != (6144, 204_800, 1_048_576) {
        bad.push(format!(
            "hidden {} vocab {} ctx {}",
            s.hidden, s.vocab, s.ctx_train
        ));
    }
    if s.rms_eps.to_bits() != hp.rms_eps.to_bits() || hp.rms_eps != 1e-5 {
        bad.push(format!("rms_eps {} (hp {:?})", s.rms_eps, hp.rms_eps));
    }
    if hp.idx_norm_eps != 1e-6 {
        bad.push(format!("indexer norm eps {}", hp.idx_norm_eps));
    }
    if !read.defaults.is_empty() {
        bad.push(format!("defaults {:?}", read.defaults));
    }
    if unread != ["general.sampling.temp", "general.sampling.top_p"] {
        bad.push(format!("unread keys {unread:?}"));
    }
    if s.chat.pre != "tekken-num3"
        || s.chat.tools.is_some()
        || s.chat.reasoning.is_some()
        || s.chat.template.as_ref().is_none_or(|t| t.len() != 14_264)
    {
        bad.push(format!(
            "chat pre {} tools {:?} reasoning {:?} template {:?}",
            s.chat.pre,
            s.chat.tools,
            s.chat.reasoning,
            s.chat.template.as_ref().map(|t| t.len())
        ));
    }
    if s.logit_epilogue
        != Some(LogitEpilogue {
            scale: 1.0,
            softcap: 30.0,
        })
    {
        bad.push(format!("logit epilogue {:?}", s.logit_epilogue));
    }
    if s.layers.len() != 72 || !s.mtp.is_empty() || s.hc.is_some() || s.engram.is_some() {
        bad.push(format!(
            "{} layers, mtp {} hc {:?} engram {:?}",
            s.layers.len(),
            s.mtp.len(),
            s.hc,
            s.engram
        ));
    }
    for (l, layer) in s.layers.iter().enumerate() {
        let models::Mixer::Latent(a) = &layer.mixer else {
            bad.push(format!("layer {l}: {:?}", layer.mixer));
            continue;
        };
        let nope = NOPE.contains(&l);
        let (pe, qk) = if nope { (0, 128) } else { (64, 64) };
        if (a.heads, a.q_lora, a.latent, a.pe) != (64, 0, 512, pe)
            || a.up != (LatentUp::Absorbed { qk, v: 128 })
            || a.out != models::LatentOut::Plain
            || a.out_gate != OutGate::None
            || a.window.is_some()
            || a.sinks
            || a.compress.is_some()
            || !a.ssmax
            || a.q_head_norm
        {
            bad.push(format!("layer {l} latent {a:?}"));
        }
        if let Some(rope) = &a.rope {
            if nope
                || rope.dims != 64
                || rope.base != 10_000.0
                || !rope.factors
                || rope.yarn.as_ref().is_none_or(|y| {
                    (y.factor, y.orig_ctx, y.beta_fast, y.beta_slow, y.magnitude)
                        != (2048.0, 512, 32.0, 1.0, 1.0)
                })
            {
                bad.push(format!("layer {l} rope {rope:?}"));
            }
        } else if !nope {
            bad.push(format!("layer {l}: no rope on a roped layer"));
        }
        let want_cache = if KV_READERS.contains(&l) {
            Source::From(35)
        } else {
            Source::Own(())
        };
        if a.cache != want_cache {
            bad.push(format!("layer {l} cache {:?}", a.cache));
        }
        let Some(Selector::StreamTopK {
            heads,
            d,
            k,
            keys,
            list,
            candidates,
        }) = a.select.as_ref()
        else {
            bad.push(format!("layer {l}: {:?}", a.select));
            continue;
        };
        let own = full(l);
        let from = if own { l } else { prev_full(l) };
        let want_keys = if own {
            Source::Own(IndexKeys::FromInput { norm_eps: 1e-6 })
        } else {
            Source::From(from as u32)
        };
        let want_list = if own {
            Source::Own(())
        } else {
            Source::From(from as u32)
        };
        if (*heads, *d, *k) != (32, 128, 2048)
            || *keys != want_keys
            || *list != want_list
            || candidates.is_some()
        {
            bad.push(format!(
                "layer {l} select {heads}x{d} k{k} keys {keys:?} list {list:?}"
            ));
        }
        if l == 0 {
            if layer.ffn
                != (Ffn::Dense {
                    ff: 24_576,
                    act: Act::SwiGlu { limit: None },
                })
            {
                bad.push(format!("layer 0 ffn {:?}", layer.ffn));
            }
            continue;
        }
        let models::Ffn::Moe(m) = &layer.ffn else {
            bad.push(format!("layer {l} ffn {:?}", layer.ffn));
            continue;
        };
        let want_moe = Moe {
            experts: 128,
            top_k: 4,
            expert_ff: 6144,
            latent: None,
            pre_exp_norm: false,
            post_exp_norm: None,
            act: Act::SwiGlu { limit: None },
            router: models::Router {
                score: Score::Softmax,
                bias: false,
                norm: RouterNorm::SelectedSum,
                scale: 1.0,
                hash: false,
            },
            shared: Some(models::Shared {
                ff: 6144,
                act: Act::SwiGlu { limit: None },
                sigmoid_gate: false,
            }),
        };
        if *m != want_moe {
            bad.push(format!("layer {l} moe {m:?}"));
        }
        if layer.residual != models::Residual::Plain || !layer.extras.is_empty() {
            bad.push(format!("layer {l} residual {:?}", layer.residual));
        }
    }
    // The tensor walk: every tensor of the header, at its role.
    let t = &read.tensors;
    if (t.layers, t.experts, t.experts_used) != (72, 128, 4) || t.tensors.len() != 1602 {
        bad.push(format!(
            "tensors' counts {} {} {} of {}",
            t.layers,
            t.experts,
            t.experts_used,
            t.tensors.len()
        ));
    }
    let tensor = |name: &str| {
        t.tensors
            .iter()
            .find(|x| x.name == name)
            .map(|x| (x.role, x.ty, x.dims.clone()))
    };
    for (name, want) in [
        (
            "token_embd.weight",
            (
                models::Role::TokenEmbedding,
                GgmlType::Q6_K,
                vec![6144, 204_800],
            ),
        ),
        (
            "output.weight",
            (models::Role::Head, GgmlType::Q6_K, vec![6144, 204_800]),
        ),
        (
            "rope_freqs.weight",
            (models::Role::Unread, GgmlType::F32, vec![32]),
        ),
        (
            "blk.0.attn_ssmax.weight",
            (models::Role::Attention, GgmlType::F32, vec![64]),
        ),
        (
            "blk.0.ffn_down.weight",
            (models::Role::DenseFfn, GgmlType::Q8_0, vec![24_576, 6144]),
        ),
        (
            "blk.1.indexer.attn_q_b.weight",
            (models::Role::Attention, GgmlType::Q6_K, vec![6144, 4096]),
        ),
        (
            "blk.17.ffn_gate_exps.weight",
            (
                models::Role::RoutedExperts,
                GgmlType::IQ1_S,
                vec![6144, 6144, 128],
            ),
        ),
        (
            "blk.35.attn_k_b.weight",
            (models::Role::Attention, GgmlType::Q8_0, vec![128, 512, 64]),
        ),
        (
            "blk.39.attn_k_b.weight",
            (models::Role::Attention, GgmlType::Q8_0, vec![128, 512, 64]),
        ),
        (
            "blk.71.indexer.proj.weight",
            (models::Role::Attention, GgmlType::F32, vec![6144, 32]),
        ),
    ] {
        if tensor(name) != Some(want) {
            bad.push(format!("{name} {:?}", tensor(name)));
        }
    }
    // The reader holds a reader layer's cache stems to the owner's: layer 39
    // carries no `attn_kv_a_mqa`, and no shared-indexer layer an indexer
    // tensor of its own.
    for (name, gone) in [
        ("blk.39.attn_kv_a_mqa.weight", true),
        ("blk.39.attn_kv_a_norm.weight", true),
        ("blk.17.indexer.attn_q_b.weight", true),
        ("blk.71.indexer.attn_q_b.weight", false),
    ] {
        if t.tensors.iter().any(|x| x.name == name) != !gone {
            bad.push(format!("{name}: present {}, wanted {}", !gone, gone));
        }
    }
    // The program is the plan's (`Program::MistralLarge4Body`) and no row
    // lists it: the coverage list names every need of the file, the new ones
    // among them — pinned here, emptied by the body's round. What no row of
    // the program runs: the two latent shapes (the absorbed query with and
    // without the rope part), SSMax, the shared cache and the keys from the
    // layer input, beside the tool parser.
    // PIN(2026-11-02): the unrotated-key row, the 32x128 stream indexer, the
    // softmax router, the dense and shared SwiGLU widths and the logit
    // epilogue joined the list when `program_of` named the family: a need is
    // covered by a row that lists the program, and these were covered by other
    // programs' rows while no program ran the file. The program's own item is
    // gone. The pre-tokenizer `tekken-num3` is a known one, so no item.
    for want in [
        "latent attention 512 x rope 64, absorbed qk 64 / v 128",
        "latent attention 512 x rope 0, absorbed qk 128 / v 128",
        "latent attention with 64 unrotated key values beside the latent, shared by every head",
        "SSMax: the query scaled per head by s*ln(pos + 1) + b",
        "the latent cache another layer wrote, read in place of its own",
        "index keys projected from the layer input, one per position",
        "stream indexer: 32 heads x 128",
        "router: softmax, 128 experts, top 4",
        "shared expert, ff 6144",
        "dense SwiGLU layer: ff 24576",
        "logit epilogue: scale 1, then the tanh softcap at 30",
        "a tool-call parser for this template",
    ] {
        if !items.iter().any(|f| f == want) {
            bad.push(format!("{want}: not in the coverage list"));
        }
    }
    // The context bound: every position the file trained (1M here); past it
    // is named. PIN(2026-11-01): the card tier of R4a lifted the fixture's
    // bound from the indexer's top-k to the trained context (the selected
    // walk serves past the top-k), so the header's own bound is the trained
    // one alone.
    let ctx = model::arch::mistrallarge4::spec::serve_ctx;
    if ctx(1_048_576, s) != Ok(1_048_576) {
        bad.push(format!("serve_ctx(1_048_576) = {:?}", ctx(1_048_576, s)));
    }
    let refused = ctx(1_048_577, s).expect_err("past the trained context");
    if !refused.to_string().contains("trained at 1048576") {
        bad.push(format!("serve_ctx(1_048_577): {refused}"));
    }

    println!(
        "mistral-large4 layers=72 nope={} kv_readers={} full_indexers={} tensors=1602 \
         template=14264 coverage_items={} pins_differ={}",
        NOPE.len(),
        KV_READERS.len(),
        (0..72).filter(|&l| full(l)).count(),
        items.len(),
        bad.len()
    );
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
        std::env::temp_dir().join(format!("bloomery-ml4-fixture-{}.gguf", std::process::id()));
    Split::header_only(path, &layouts[0].1).expect("the fixture's header-only split opens")
}

/// The fixture clause: the plan of the fixture of the real header's five layers `[0, 1, 17, 35,
/// 39]` (`model::arch::mistrallarge4::fixture`), laid out as a header-only file, reads through the
/// engine's own reader as the cut. The three per-layer arrays are the source's at the map's layers
/// (derived here from the pins above, not from the fixture's own table): `rope_pattern`
/// `[T,T,T,F,F]`, `indexer.types` `[T,T,F,T,T]` and the cache map in fixture layers
/// `[0,1,2,3,3]`; the mini's widths are its keys' and every tensor holds the real layer's type. A
/// map whose layers are other kinds, an array left at the source's values, a mini key left at the
/// source's and a retyped tensor are each refused by name.
#[test]
fn mistrallarge4_fixture_cut() {
    let source = header();
    let spec = ml4::spec();
    let plan = model::fixture::plan(&spec, &source, None, &spec.options())
        .unwrap_or_else(|e| panic!("plan: {e}"));
    let fx = fixture_header(&plan.target);
    let hp = model::arch::mistrallarge4::hparams::Hparams::read(&fx)
        .unwrap_or_else(|e| panic!("the fixture's hparams: {e}"));
    let read = model::arch::spec(&fx).unwrap_or_else(|e| panic!("the fixture's spec: {e}"));
    let s = &read.spec;
    let mut bad = Vec::new();

    // The arrays, from the pins of the real header: a NoPE layer ropes not, a shared indexer is
    // the pairs from 17, and a KV reader reads layer 35's cache.
    let cut = ml4::LAYER_MAP;
    let fixture_layer = |l: usize| cut.iter().position(|&m| m == l).expect("a map layer");
    let want_rope: Vec<bool> = cut.iter().map(|l| !NOPE.contains(l)).collect();
    let want_full: Vec<bool> = cut.iter().map(|&l| full(l)).collect();
    let want_cache: Vec<usize> = cut
        .iter()
        .map(|&l| fixture_layer(if KV_READERS.contains(&l) { 35 } else { l }))
        .collect();
    if (&hp.rope_layer, &hp.indexer_full, &hp.kv_source) != (&want_rope, &want_full, &want_cache) {
        bad.push(format!(
            "arrays rope {:?} indexer {:?} cache {:?}, the cut's {want_rope:?} {want_full:?} {want_cache:?}",
            hp.rope_layer, hp.indexer_full, hp.kv_source
        ));
    }
    if (
        want_rope.as_slice(),
        want_full.as_slice(),
        want_cache.as_slice(),
    ) != (
        [true, true, true, false, false].as_slice(),
        [true, true, false, true, true].as_slice(),
        [0, 1, 2, 3, 3].as_slice(),
    ) {
        bad.push(format!(
            "the cut's arrays {want_rope:?} {want_full:?} {want_cache:?} are not the five kinds"
        ));
    }

    // The mini's widths and the facts kept.
    let got = (
        hp.n_layer,
        hp.n_embd,
        hp.n_head,
        hp.n_expert,
        hp.expert_ff,
        hp.dense_ff,
        hp.indexer_top_k,
    );
    if got != (5, 1024, 16, 128, 256, 4096, 16) {
        bad.push(format!("mini widths {got:?}"));
    }
    let kept = (
        hp.kv_lora,
        hp.key_length,
        hp.head_k,
        hp.head_v,
        hp.rope_dims,
        hp.indexer_heads,
        hp.indexer_d,
        hp.n_used,
        hp.n_shared,
        hp.n_vocab,
    );
    if kept != (512, 576, 128, 128, 64, 32, 128, 4, 1, 204_800) {
        bad.push(format!("kept facts {kept:?}"));
    }

    // The layers the reader describes: every one selects a top-k of 16 over its positions, the
    // shared indexer reads layer 1's keys and list, layer 4 reads layer 3's cache.
    for (f, layer) in s.layers.iter().enumerate() {
        let models::Mixer::Latent(a) = &layer.mixer else {
            bad.push(format!("layer {f}: {:?}", layer.mixer));
            continue;
        };
        match a.select.as_ref() {
            Some(Selector::StreamTopK { k: 16, list, .. }) => {
                let want_list = if want_full[f] {
                    Source::Own(())
                } else {
                    Source::From(1)
                };
                if *list != want_list {
                    bad.push(format!("layer {f} list {list:?}"));
                }
            }
            other => bad.push(format!("layer {f} select {other:?}")),
        }
        let cache = if f == 4 {
            Source::From(3)
        } else {
            Source::Own(())
        };
        if a.cache != cache {
            bad.push(format!("layer {f} cache {:?}", a.cache));
        }
    }
    // The reader serves every position the fixture trained, past the
    // indexer's own top-k 16 among them; past the trained one is named. The
    // cut copies the source's context_length (key_rule's COPY), 1M.
    // PIN(2026-11-01): R4a's card tier lifted the top-k bound (above); the
    // trained context is the copied 1M, not the mini's 64.
    let ctx = model::arch::mistrallarge4::spec::serve_ctx;
    let trained = 1_048_576;
    if ctx(16, s) != Ok(trained) || ctx(trained, s) != Ok(trained) || ctx(trained + 1, s).is_ok() {
        bad.push(format!(
            "serve_ctx(16) = {:?}, ({trained}) = {:?}, ({}) = {:?}",
            ctx(16, s),
            ctx(trained, s),
            trained + 1,
            ctx(trained + 1, s)
        ));
    }

    // Every tensor at the real layer's type, the stems of each kind.
    let t = &read.tensors;
    if t.tensors.len() != 119 {
        bad.push(format!(
            "{} tensors, the five layers hold 119",
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
        ("token_embd.weight", (GgmlType::Q6_K, vec![1024, 204_800])),
        ("rope_freqs.weight", (GgmlType::F32, vec![32])),
        ("blk.0.attn_k_b.weight", (GgmlType::Q8_0, vec![64, 512, 16])),
        ("blk.0.ffn_down.weight", (GgmlType::Q8_0, vec![4096, 1024])),
        (
            "blk.1.ffn_down_exps.weight",
            (GgmlType::IQ2_XXS, vec![256, 1024, 128]),
        ),
        (
            "blk.2.ffn_gate_exps.weight",
            (GgmlType::IQ1_S, vec![1024, 256, 128]),
        ),
        (
            "blk.2.ffn_down_exps.weight",
            (GgmlType::IQ2_XXS, vec![256, 1024, 128]),
        ),
        (
            "blk.3.attn_k_b.weight",
            (GgmlType::Q8_0, vec![128, 512, 16]),
        ),
        (
            "blk.3.attn_kv_a_mqa.weight",
            (GgmlType::Q8_0, vec![1024, 512]),
        ),
        (
            "blk.3.ffn_down_exps.weight",
            (GgmlType::IQ1_S, vec![256, 1024, 128]),
        ),
        (
            "blk.4.indexer.attn_q_b.weight",
            (GgmlType::Q6_K, vec![1024, 4096]),
        ),
    ] {
        if tensor(name) != Some(want.clone()) {
            bad.push(format!("{name} {:?}, wanted {want:?}", tensor(name)));
        }
    }
    for name in [
        "blk.4.attn_kv_a_mqa.weight",
        "blk.4.attn_kv_a_norm.weight",
        "blk.2.indexer.attn_q_b.weight",
    ] {
        if tensor(name).is_some() {
            bad.push(format!("{name} is in the fixture"));
        }
    }
    if let Err(e) = ml4::check_kinds(&spec, &fx, &source) {
        bad.push(format!("check_kinds: {e}"));
    }

    // The refusals, each by name.
    let mutant = |what: &str, want: &str, edit: &dyn Fn(&mut FilePlan)| {
        let mut p = plan.target.clone();
        edit(&mut p);
        let split = fixture_header(&p);
        match ml4::check_kinds(&spec, &split, &source) {
            Err(e) if e.to_string().contains(want) => Ok(()),
            Err(e) => Err(format!("{what}: refused as {e}, wanted {want:?}")),
            Ok(_) => Err(format!("{what}: accepted")),
        }
    };
    let set = |p: &mut FilePlan, key: &str, v: Value| {
        p.kvs
            .iter_mut()
            .find(|(k, _)| k == &format!("mistral-large4.{key}"))
            .unwrap_or_else(|| panic!("{key} is in the plan"))
            .1 = v;
    };
    let bools = |v: &[bool]| Value::Array(v.iter().map(|&b| Value::Bool(b)).collect());
    let results = [
        mutant(
            "an indexer array left at the source's",
            "blk.2.indexer.q_norm.weight",
            &|p| {
                set(
                    p,
                    "attention.indexer.types",
                    bools(&[true, true, true, true, true]),
                );
            },
        ),
        mutant(
            "the cache map left at the source's ids",
            "attention.kv_source_map",
            &|p| {
                let ids = [0u32, 1, 17, 35, 35];
                set(
                    p,
                    "attention.kv_source_map",
                    Value::Array(ids.iter().map(|&i| Value::U32(i)).collect()),
                );
            },
        ),
        mutant(
            "the top-k left at the source's",
            "attention.indexer.top_k",
            &|p| {
                set(p, "attention.indexer.top_k", Value::U32(2048));
            },
        ),
        mutant(
            "a tensor of another type",
            "ffn_gate_exps.weight is",
            &|p| {
                let t = p
                    .tensors
                    .iter_mut()
                    .find(|t| t.name == "blk.2.ffn_gate_exps.weight")
                    .expect("the stack is in the plan");
                t.ty = GgmlType::IQ2_XXS;
                let (_, blck, tsz) = gguf::ggml_type_info(t.ty.as_u32()).expect("a type");
                t.nbytes = gguf::tensor_bytes(blck, tsz, &t.dims).expect("bytes");
            },
        ),
    ];
    for r in results {
        if let Err(why) = r {
            bad.push(why);
        }
    }
    // A map of other kinds: layer 2 is a full indexer, the kind keeps it shared.
    let other = FixtureSpec {
        layers: vec![0, 1, 2, 35, 39],
        ..ml4::spec()
    };
    match model::fixture::plan(&other, &source, None, &other.options()) {
        Err(e)
            if e.to_string()
                .contains("source layer 2 (fixture 2) has a full indexer") => {}
        Err(e) => bad.push(format!("a map of other kinds: refused as {e}")),
        Ok(_) => bad.push("a map of other kinds: planned".into()),
    }

    println!(
        "mistral-large4 fixture layers=5 tensors={} shards={} pins_differ={}",
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

/// The Mix-256G file's header tables (its first shard's keys, every tensor of its six shards),
/// from the dump the quantizer's README describes: the same names and dims as the UD-IQ1_S file,
/// 33 layers whose gate and up stacks differ in type.
const MIX_KV: &str = include_str!("mistrallarge4_mix256_hdr.kv.json");
const MIX_TENSORS: &str = include_str!("mistrallarge4_mix256_tensors.tsv");

/// The positions the plans are made at, and the bytes a position holds. The cache is the layers'
/// own rows and keys, derived from the header's three per-layer arrays: a layer that writes its
/// cache holds one f16 row of the latent and the rope part (512 + 64 values on a roped layer, 512
/// on a NoPE one), a KV reader none, and a layer that runs its own indexer one f16 key of 128
/// values (256 B): 54 × 1,152 + 9 × 1,024 + 44 × 256 = 82,688 B.
const PLAN_CTX: u64 = 4096;
const KV_PER_POSITION: u64 = 82_688;

/// The types of the experts' stacks the common card leg runs: every stack of both files is one.
const LEG_TYPES: [GgmlType; 4] = [
    GgmlType::IQ1_S,
    GgmlType::IQ1_M,
    GgmlType::IQ2_XXS,
    GgmlType::IQ2_XS,
];

/// What a file's plan is held to: the layers the rule keeps on the host (a gate and an up of two
/// types), the band of the prompt batch's reserve, the reserve in bytes priced for the lever's
/// default prompt group, and the bands of the split in bytes, in tenths of a GB: the gate plan's
/// card experts and host experts, then plan (b′)'s.
struct PlanWant {
    host_only: usize,
    reserve: (u64, u64),
    group_reserve: u64,
    bands: [(u64, u64); 4],
}

/// A tenth of a GB, the bands' unit.
const DGB: u64 = 100_000_000;

/// One card of a plan against the terms the header derives. The card's terms are its dense bytes
/// (everything but the routed stacks, the token embedding and the rope factors, which the plan
/// keeps on the host), its cache, the machine's context and scratch; its allocator rounding is the
/// allocator model over the device buffers `uploads` of its segments in the loader's order
/// (`allocator_bytes`: a buffer of a granule or more takes whole granules, smaller ones share
/// them); the load makes `reserve` bytes more on every card, the prompt batch's buffers and on a
/// machine with an expert tier card the tier's places (`PlanInputs::prompt_batch_bytes`), which
/// the plan keeps out of the expert budget and out of `CardTotals`; and the expert rule adds one
/// expert at a time while the card fits `usable - margin` beside all of them, so what is left
/// under the margin is less than one expert (`expert_max` bytes). Pushes the pins that differ onto
/// `bad` and returns the bytes left.
fn card_terms_of(
    label: &str,
    card: &model::placement::CardTotals,
    spec: &model::placement::workstation::CardSpec,
    (dense, kv, reserve): (u64, u64, u64),
    uploads: &[u64],
    expert_max: u64,
    bad: &mut Vec<String>,
) -> u64 {
    use model::placement::allocator_bytes;
    use model::placement::workstation::{CONTEXT, SCRATCH};
    let rounding =
        allocator_bytes(spec.granule_bytes, uploads.iter().copied()) - uploads.iter().sum::<u64>();
    if (
        card.dense_bytes,
        card.kv_bytes,
        card.stage_rows_bytes,
        card.context_bytes,
        card.scratch_bytes,
        card.rounding_bytes,
    ) != (dense, kv, 0, CONTEXT, SCRATCH, rounding)
    {
        bad.push(format!(
            "{label}: dense {} kv {} stage rows {} context {} scratch {} rounding {}, derived \
             {dense} {kv} 0 {CONTEXT} {SCRATCH} {rounding}",
            card.dense_bytes,
            card.kv_bytes,
            card.stage_rows_bytes,
            card.context_bytes,
            card.scratch_bytes,
            card.rounding_bytes
        ));
    }
    let taken = card.dense_bytes
        + card.expert_bytes
        + card.rounding_bytes
        + card.kv_bytes
        + card.stage_rows_bytes
        + card.scratch_bytes
        + card.context_bytes
        + card.reserve_bytes
        + reserve;
    let limit = spec.usable_bytes() - spec.margin();
    let left = limit.saturating_sub(taken);
    if taken > limit || left >= expert_max {
        bad.push(format!(
            "{label}: {left} B left under the margin after {taken} B (the load's {reserve} B \
             among them), the rule stops within one expert ({expert_max} B)"
        ));
    }
    left
}

/// A file's plans on the gate plan (the 3090's bytes) and plan (b′) (the A6000 the stage, the 3090 an
/// expert tier) at [`PLAN_CTX`] positions, held to what the header derives; prints the planned
/// split. The pins that differ, every one.
fn plan_pins(tag: &str, kv: &str, tensors: &str, want: &PlanWant) -> Vec<String> {
    use model::arch::mistrallarge4::place::{
        PROMPT_ROWS, PlanInputs, configure_group, prompt_reserve_bytes, tier_batch,
    };
    use model::placement::workstation::{self, HostRead};
    use model::placement::{KvBytes, Machine, PlanLevers};

    let mut bad = Vec::new();
    let split = header_of(tag, kv, tensors);
    let mut inputs =
        PlanInputs::describe(&split).unwrap_or_else(|e| panic!("{tag}: describe: {e}"));
    // The box's planning room, so a plan does not read the machine that runs the test.
    inputs.room = (workstation::PLAN_ROOM, HostRead::Given);

    // The header's own bytes: the table's, never the plan's.
    let rows: Vec<(TensorDecl, GgmlType)> = tensors.lines().map(decl).collect();
    let layer_of = |name: &str| -> usize {
        name.split('.')
            .nth(1)
            .and_then(|l| l.parse().ok())
            .expect("a layer")
    };
    let is_stack = |name: &str| name.contains("ffn_") && name.contains("_exps.");
    let (mut routed, mut embd, mut unread, mut total) = (0, 0, 0, 0);
    let mut stacks = std::collections::BTreeMap::<usize, [(GgmlType, u64); 3]>::new();
    for (d, ty) in &rows {
        total += d.nbytes;
        if d.name == "token_embd.weight" {
            embd = d.nbytes;
        } else if d.name == "rope_freqs.weight" {
            // `Role::Unread`: the factors are the host's, no card holds them.
            unread = d.nbytes;
        } else if is_stack(&d.name) {
            routed += d.nbytes;
            let part = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
                .iter()
                .position(|p| d.name.contains(p))
                .expect("a stack");
            stacks
                .entry(layer_of(&d.name))
                .or_insert([(GgmlType::F32, 0); 3])[part] = (*ty, d.nbytes);
        }
    }
    let dense = total - routed - embd - unread;
    let expert_bytes_of = |l: usize| stacks[&l].iter().map(|&(_, b)| b).sum::<u64>() / 128;
    let host_only: Vec<usize> = stacks
        .iter()
        .filter(|(_, s)| s[0].0 != s[1].0)
        .map(|(&l, _)| l)
        .collect();
    let leg_types = stacks
        .values()
        .flatten()
        .all(|(ty, _)| LEG_TYPES.contains(ty));
    let expert_max = stacks
        .keys()
        .filter(|l| !host_only.contains(l))
        .map(|&l| expert_bytes_of(l))
        .max()
        .expect("a card layer");
    if !leg_types || host_only.len() != want.host_only {
        bad.push(format!(
            "stacks: leg types {leg_types}, {} layers of two gate·up types, wanted {}",
            host_only.len(),
            want.host_only
        ));
    }

    // The cache: the layers' bytes a position, summed over the trunk, as the arrays derive them.
    let layer_kv = |l: usize| {
        let cache = match (KV_READERS.contains(&l), NOPE.contains(&l)) {
            (true, _) => 0,
            (false, true) => 1024,
            (false, false) => 1152,
        };
        cache + if full(l) { 256 } else { 0 }
    };
    let derived_kv: u64 = (0..72).map(layer_kv).sum();
    let planned_kv = inputs.kv.bytes(0..72, 1);
    let per_layer: Vec<usize> = (0..72)
        .filter(|&l| inputs.kv.layer_bytes(l, 1) != layer_kv(l))
        .collect();
    if derived_kv != KV_PER_POSITION || planned_kv != KV_PER_POSITION || !per_layer.is_empty() {
        bad.push(format!(
            "kv per position: planned {planned_kv} B, derived {derived_kv} B, wanted \
             {KV_PER_POSITION} B (layers that differ {per_layer:?})"
        ));
    }
    let kv = KV_PER_POSITION * PLAN_CTX;
    let levers = PlanLevers::default();

    // The gate plan and plan (b′), each card against the derived terms.
    let gate = workstation::plan_gate(72);
    let batch = tier_batch(&inputs).unwrap_or_else(|e| panic!("{tag}: tier batch: {e}"));
    let bp = workstation::plan_bp(72, None, batch);

    // The bytes the load makes beside the plan's terms on every card, which the expert rule keeps
    // out of the budget: the prompt batch's buffers (the family's formula at the plan's block of
    // rows and one unit, banded by the file's types) and, where an expert tier card is named, the
    // tier's places of a block, 12 B for each of its slots.
    let reserve_of = |machine: &Machine| {
        inputs
            .prompt_batch_bytes(machine)
            .unwrap_or_else(|e| panic!("{tag}: prompt batch bytes: {e}"))
    };
    let (reserve_gate, reserve_bp) = (reserve_of(&gate), reserve_of(&bp));
    let formula = prompt_reserve_bytes(
        &inputs.hp,
        &inputs.spec,
        &inputs.model,
        PROMPT_ROWS as u64,
        1,
    )
    .unwrap_or_else(|e| panic!("{tag}: prompt reserve: {e}"));
    let places = 12 * PROMPT_ROWS as u64 * inputs.hp.n_used as u64;
    let (low, high) = want.reserve;
    if inputs.prompt_reserve != formula
        || reserve_gate != formula
        || reserve_bp != formula + places
        || formula < low * DGB
        || formula > high * DGB
    {
        bad.push(format!(
            "prompt batch: the plan's {} B, the formula's {formula} B, {reserve_gate} B on the \
             gate plan and {reserve_bp} B on plan (b′), wanted the formula's (inside [{low}, \
             {high}] tenths of a GB) and {places} B of tier places more on (b′)",
            inputs.prompt_reserve
        ));
    }

    // The prompt group: `configure_group` prices every unit a group holds past the first at its
    // record and list (live and selected counts, walk pairs, rope rows and their bits, `top_k`
    // ids a row) and its walk unit (x, x1 and the positions), read off the header's widths
    // alone [derived]; a group outside 1 to the lever's most is refused and leaves the reserve.
    let (block, shape) = (PROMPT_ROWS as u64, &inputs.hp);
    let record = 4
        * (4 * block + 1 + 2 * block * shape.rope_dims as u64 + block * shape.indexer_top_k as u64);
    let walk_unit = block * (8 * shape.n_embd as u64 + 8);
    let default_group = bloomery_levers::PREFILL_GROUP_DEFAULT as usize;
    let most_group = bloomery_levers::PREFILL_GROUP_MAX as usize;
    for group in [1, default_group, most_group] {
        let derived =
            formula + (runtime::prompt::group_sets(group) as u64 - 1) * (record + walk_unit);
        let mut grouped = inputs.clone();
        let priced = configure_group(&mut grouped, group)
            .map(|()| grouped.prompt_reserve)
            .map_err(|e| e.to_string());
        if !matches!(priced, Ok(bytes) if bytes == derived
            && (group != default_group || bytes == want.group_reserve))
        {
            bad.push(format!(
                "prompt group {group}: configure_group priced {priced:?} B, derived {derived} B \
                 (one unit {formula} B and {} more of a record {record} B and a walk unit \
                 {walk_unit} B), wanted {} B at the default group {default_group}",
                runtime::prompt::group_sets(group) - 1,
                want.group_reserve
            ));
        }
    }
    for group in [0, most_group + 1] {
        let mut grouped = inputs.clone();
        if configure_group(&mut grouped, group).is_ok()
            || grouped.prompt_reserve != inputs.prompt_reserve
        {
            bad.push(format!(
                "prompt group {group}: configure_group took a group outside 1 to {most_group} or \
                 changed the reserve"
            ));
        }
    }

    let mut splits = Vec::new();
    for (label, machine, specs, reserve) in [
        ("gate", &gate, vec![workstation::RTX_3090], reserve_gate),
        (
            "bp",
            &bp,
            vec![workstation::A6000, workstation::RTX_3090],
            reserve_bp,
        ),
    ] {
        let plan = inputs
            .plan(machine, PLAN_CTX, &levers)
            .unwrap_or_else(|e| panic!("{tag}: plan {label}: {e}"));
        let (mut cards, mut left) = (0, Vec::new());
        for (c, (card, spec)) in plan.cards.iter().zip(&specs).enumerate() {
            // The card's uploads in the plan's row order, the tensor order: the device buffers of
            // each segment it holds (a Q8_0 tensor is two planes, a routed stack its experts').
            let uploads: Vec<u64> = plan
                .rows
                .iter()
                .flat_map(|r| {
                    let t = &plan.model.tensors[r.tensor];
                    r.segments
                        .iter()
                        .filter(|s| s.device == model::placement::Device::Card(c))
                        .flat_map(|s| {
                            s.buffer_bytes(t, plan.model.experts)
                                .unwrap_or_else(|e| panic!("{tag} {label}: {}: {e}", t.name))
                        })
                        .collect::<Vec<u64>>()
                })
                .collect();
            let derived = if c == 0 {
                (dense, kv, reserve)
            } else {
                (0, 0, reserve)
            };
            left.push(card_terms_of(
                &format!("{tag} {label} card {c}"),
                card,
                spec,
                derived,
                &uploads,
                expert_max,
                &mut bad,
            ));
            cards += card.expert_bytes;
        }
        let host = plan.host.expert_bytes;
        if cards + host + plan.host.nvme_expert_bytes != routed || plan.host.nvme_expert_bytes != 0
        {
            bad.push(format!(
                "{tag} {label}: card {cards} + host {host} + nvme {} B, the table's routed \
                 {routed} B with none paged",
                plan.host.nvme_expert_bytes
            ));
        }
        let held: Vec<usize> = (0..72)
            .filter(|&l| plan.n_l[l] > 0 || plan.tier_n_l.iter().any(|t| t[l] > 0))
            .collect();
        if let Some(l) = held.iter().find(|l| host_only.contains(l)) {
            bad.push(format!(
                "{tag} {label}: layer {l} holds card experts, its gate and up differ in type"
            ));
        }
        println!(
            "{tag} plan {label} @ ctx {PLAN_CTX}: kv {kv} B ({KV_PER_POSITION} B/token), card \
             experts {cards} B ({:?} per card), host experts {host} B, dense on the stage {} B, \
             rounding {:?} B, prompt batch {reserve} B, left under the margin {left:?} B, {} of {} \
             card layers hold card experts, {} layers host-only by the rule (gate and up of two \
             types) {host_only:?}",
            plan.cards
                .iter()
                .map(|c| c.expert_bytes)
                .collect::<Vec<_>>(),
            plan.cards[0].dense_bytes,
            plan.cards
                .iter()
                .map(|c| c.rounding_bytes)
                .collect::<Vec<_>>(),
            held.len(),
            stacks.len() - host_only.len(),
            host_only.len(),
        );
        splits.push((cards, host));
    }
    let got = [splits[0].0, splits[0].1, splits[1].0, splits[1].1];
    for (what, (got, (lo, hi))) in ["gate card", "gate host", "bp card", "bp host"]
        .iter()
        .zip(got.into_iter().zip(want.bands))
    {
        if got < lo * DGB || got > hi * DGB {
            bad.push(format!(
                "{tag} {what} experts {got} B outside [{lo}, {hi}] tenths of a GB"
            ));
        }
    }
    bad
}

/// The UD-IQ1_S header planned on the gate plan and on plan (b′): the derived split. The cache is
/// 82,688 B a token (the 63 layers that write a cache hold their rows, the 9 readers none, the 44
/// indexer layers their keys), 0.34 GB at the plan's 4,096 positions. The stage keeps the 17.97 GB
/// of dense bytes (the file's 18.99 GB less the embedding the plan leaves on the host), so the gate
/// plan's 25.35 GB card, less the 0.54 GB context, 0.07 GB scratch, 1.07 GB margin, the cache and
/// the 0.78 GB the prompt batch's buffers take (`PlanInputs::prompt_reserve`, kept out of every
/// card's expert budget), leaves 4.59 GB for experts and the allocator's rounding [derived]; the
/// rounding is the 1.00 GB the allocator model takes in whole granules for buffers of a granule or
/// more (the stage's 1,387 dense tensors, Q8_0 ones as two planes), so the card holds 3.57 GB of
/// experts and the host 210.4 GB. Plan (b′)'s stage holds 29.36 GB and the tier 22.56 GB (0.25 GB
/// of rounding), 51.9 GB on the cards and 162.1 GB on the host. The design's figures were 5.8 GB
/// on the gate card (before the cache and the rounding, 1.3 GB over), 208 GB, 54 GB and 160 GB.
/// Bands: the planned bytes ±0.1 GB, held by the derived terms above. Mutant: `KvLayout::of`
/// counting a KV reader's rows (the cache is 9 × 1,152 B a token over) or none of the index keys,
/// or `plan_with_slots` keeping more out of the expert budget than the load makes (each card's
/// left under the margin passes one expert).
#[test]
fn mistrallarge4_plan_ud_iq1_s() {
    // PIN(2026-11-02): the bands fall by the prompt batch's reserve and the rounding it adds [derived]:
    // the plan keeps the family's formula at 512 rows and one unit, 783,054,324 B (the attention's
    // 302,671,188 B, the walk's 37,752,832 B, the routed block's 442,630,304 B), out of every card's
    // budget, which the bands (4.48 GB of card experts, 53.4 GB on plan (b′)) did not count; the gate
    // card holds 0.91 GB less (0.78 GB the reserve, 0.13 GB the rounding) and each plan (b′) card 0.7 GB.
    let bad = plan_pins(
        "ud-iq1_s",
        KV,
        TENSORS,
        &PlanWant {
            host_only: 0,
            reserve: (7, 9),
            // PIN(2026-11-02): the reserve at the default group, 2, which holds 3 units: the 783,054,324 B of
            // one unit and two more of a record (4,464,644 B) and a walk unit (25,169,920 B), 842,323,452 B
            // [derived], the bytes the box's load made at that group (`gate-gpu-mistrallarge4-residency`).
            group_reserve: 842_323_452,
            bands: [(35, 37), (2103, 2105), (518, 520), (1620, 1622)],
        },
    );
    assert!(
        bad.is_empty(),
        "{} pin(s) differ:\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}

/// The Mix-256G header, the same plans: 33 of the 71 routed layers have a gate and an up of two
/// types (IQ1_S with IQ1_M, IQ2_XXS with IQ1_S, ...), which the common leg's rule keeps on the host
/// whole (a layer's card format is one type for gate and up), so the 38 layers whose three stacks
/// the leg runs share the card's budget, all 38 holding card experts. The file's non-routed bytes are
/// 11.65 GB (the quantizer spent them on the routed stacks), so the stage keeps 11.1 GB dense, 6.9 GB
/// less than the UD file's, and the prompt batch's buffers take 0.83 GB of every card (the shared
/// and dense blocks are read in more forms here): the gate card holds 10.49 GB of experts and the
/// host 210.8 GB, plan (b′)'s cards 58.73 GB (stage 36.09 GB, tier 22.64 GB) and the host
/// 162.5 GB. No design figure: the terms are the header's alone, the bands the planned bytes
/// ±0.1 GB.
#[test]
fn mistrallarge4_plan_mix256() {
    // PIN(2026-11-02): the bands fall by the prompt batch's reserve and the rounding it adds [derived]:
    // the plan keeps the family's formula at 512 rows and one unit, 825,030,132 B (the attention's
    // 302,671,188 B, the walk's 37,752,832 B, the routed block's 484,606,112 B), out of every card's
    // budget, which the bands (11.41 GB of card experts, 60.5 GB on plan (b′)) did not count; the gate
    // card holds 0.92 GB less and each plan (b′) card 0.9 GB.
    let bad = plan_pins(
        "mix256",
        MIX_KV,
        MIX_TENSORS,
        &PlanWant {
            host_only: 33,
            reserve: (7, 9),
            // PIN(2026-11-02): the reserve at the default group, 2, which holds 3 units: the 825,030,132 B of
            // one unit and two more of 29,634,564 B (the UD file's, the widths are the same) [derived].
            group_reserve: 884_299_260,
            bands: [(104, 106), (2107, 2109), (586, 588), (1624, 1626)],
        },
    );
    assert!(
        bad.is_empty(),
        "{} pin(s) differ:\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}

/// The UD-IQ1_S header planned to stream its prompt calls' host experts (`PlanInputs::with_stream`):
/// the ring is two halves of 96 slots, a slot the widest layer's three stacks per expert, and the
/// ring pass's route GEMM buffers over a half at the plan's prompt rows. The header's table gives
/// the slot: the seven IQ2_XXS layers' 3 × 6144 × 6144 values at 66 B a 256-value block, 29,196,288 B
/// an expert (the 42 IQ1_S layers' 22,118,400 B and the 22 of IQ1_S gate·up over an IQ2_XXS down's
/// 24,477,696 B are under it). Both stack types read the grouped GEMM's q8_1 blocks of 128
/// (`kernels::gemm_reads_blocks32` is false for IQ1_S and IQ2_XXS), so the pass is priced in that
/// form for the gate·up rows and the down rows. The plan on the A6000 alone then keeps the ring out of
/// its card experts: they shrink by the ring's bytes to within a few experts (the rule adds one
/// expert at a time, the allocator takes whole granules). Mutant: a half of 95 or 97, the slot left
/// 0, or the pass left out of the ring's bytes.
#[test]
fn mistrallarge4_stream_plan_ud_iq1_s() {
    use model::arch::mistrallarge4::place::{PROMPT_ROWS, PlanInputs};
    use model::arch::mistrallarge4::stream::{STREAM_HALF_SLOTS, stream_bytes};
    use model::arch::place_hybrid::{RingForms, ring_pass_bytes_in};
    use model::placement::PlanLevers;
    use model::placement::workstation::{self, HostRead};

    const HALF: u64 = 96;
    const WIDEST_EXPERT: u64 = 29_196_288;
    let mut bad = Vec::new();
    let split = header_of("ud-stream", KV, TENSORS);
    let mut inputs =
        PlanInputs::describe(&split).unwrap_or_else(|e| panic!("ud-iq1_s: describe: {e}"));
    // The box's planning room, so the plan does not read the machine that runs the test.
    inputs.room = (workstation::PLAN_ROOM, HostRead::Given);

    // The header's own bytes: each layer's three stacks over the 128 experts they hold.
    let mut layers = std::collections::BTreeMap::<usize, u64>::new();
    for line in TENSORS.lines() {
        let (d, _) = decl(line);
        let stack = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
            .iter()
            .any(|p| d.name.contains(p));
        if stack {
            let l: usize = d
                .name
                .split('.')
                .nth(1)
                .and_then(|l| l.parse().ok())
                .expect("a layer");
            *layers.entry(l).or_default() += d.nbytes;
        }
    }
    let slot = layers
        .values()
        .map(|b| b / 128)
        .max()
        .expect("a routed layer");
    if slot != WIDEST_EXPERT || STREAM_HALF_SLOTS != HALF {
        bad.push(format!(
            "slot {slot} B (wanted {WIDEST_EXPERT}), half {STREAM_HALF_SLOTS} slots (wanted {HALF})"
        ));
    }
    let hp = &inputs.hp;
    let pass = ring_pass_bytes_in(
        PROMPT_ROWS,
        hp.n_used,
        hp.n_embd,
        hp.expert_ff,
        HALF as usize,
        RingForms {
            gate_up32: false,
            down32: false,
        },
    ) as u64;
    let ring = 2 * HALF * WIDEST_EXPERT + pass;
    let streamed = inputs
        .clone()
        .with_stream()
        .unwrap_or_else(|e| panic!("ud-iq1_s: with_stream: {e}"));
    let planned = stream_bytes(&inputs).unwrap_or_else(|e| panic!("ud-iq1_s: stream_bytes: {e}"));
    if planned != ring
        || streamed.stream_slot != WIDEST_EXPERT
        || streamed.prompt_reserve != inputs.prompt_reserve + ring
    {
        bad.push(format!(
            "ring {planned} B, slot {} B, reserve {} B over {}: derived ring {ring} B (pass {pass} \
             B), slot {WIDEST_EXPERT} B",
            streamed.stream_slot, streamed.prompt_reserve, inputs.prompt_reserve
        ));
    }

    // The plan on the A6000 alone: the card experts give up the ring's bytes.
    let machine = workstation::plan_a(72);
    let levers = PlanLevers::default();
    let plain = inputs
        .plan(&machine, PLAN_CTX, &levers)
        .unwrap_or_else(|e| panic!("ud-iq1_s: plan a: {e}"));
    let with = streamed
        .plan(&machine, PLAN_CTX, &levers)
        .unwrap_or_else(|e| panic!("ud-iq1_s: plan a with the ring: {e}"));
    let (a, b) = (plain.cards[0].expert_bytes, with.cards[0].expert_bytes);
    let gave = a.saturating_sub(b);
    println!(
        "ud-iq1_s plan a @ ctx {PLAN_CTX}: card experts {a} B plain, {b} B with the ring of {ring} B \
         (pass {pass} B), gave up {gave} B; the smallest card layer holds {:?} experts",
        with.n_l.iter().copied().filter(|&n| n > 0).min()
    );
    if gave + 4 * WIDEST_EXPERT < ring || gave > ring + 4 * WIDEST_EXPERT {
        bad.push(format!(
            "plan a: the card experts gave up {gave} B for a ring of {ring} B, more than four \
             experts apart"
        ));
    }
    assert!(
        bad.is_empty(),
        "{} pin(s) differ:\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}

/// A file's residency over its plans on `a`, the gate plan and plan (b′), held to what the header
/// derives. The churn pool is the stage card's experts past the first `pinned` of each layer
/// (`ChurnPool::of`, card 0): its bytes are the header's expert of each layer (its three stacks'
/// share of one id) times the plan's card experts that layer keeps past them, whatever the plan puts
/// on a tier card. A layer the plan gives no card expert (a Mix-256G layer whose gate and up
/// differ in type) has no run in the pool. The unset word on the plans of a serving placement is
/// the serving word as it is (the plan holds a card and a host expert, its fewest card experts a
/// layer hold the word's spare and the expert that moves, and its host headroom takes the pool),
/// and `off` on the gate placement; the pool at the serving word is checked against the plan's
/// headroom. Pushes the pins that differ onto `bad`.
fn residency_pins(tag: &str, kv: &str, tensors: &str, host_only: usize) -> Vec<String> {
    use bloomery_levers::{
        ResidencyAt, ResidencyWhy, UnhostedPool, residency_at_plan, residency_unset,
    };
    use model::arch::mistrallarge4::place::{PlanInputs, tier_batch};
    use model::placement::PlanLevers;
    use model::placement::churn::ChurnPool;
    use model::placement::workstation::{self, HostRead};
    use std::collections::{BTreeMap, BTreeSet};

    let mut bad = Vec::new();
    let split = header_of(tag, kv, tensors);
    let mut inputs =
        PlanInputs::describe(&split).unwrap_or_else(|e| panic!("{tag}: describe: {e}"));
    // The box's planning room, so a plan does not read the machine that runs the test.
    inputs.room = (workstation::PLAN_ROOM, HostRead::Given);

    // The header's own bytes: an expert of each layer is its three stacks' share of one id.
    let mut per_expert = BTreeMap::<usize, u64>::new();
    let mut types = BTreeMap::<usize, [Option<GgmlType>; 2]>::new();
    for line in tensors.lines() {
        let (d, ty) = decl(line);
        let Some(part) = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
            .iter()
            .position(|p| d.name.contains(p))
        else {
            continue;
        };
        let l: usize = d
            .name
            .split('.')
            .nth(1)
            .and_then(|l| l.parse().ok())
            .expect("a layer");
        if d.nbytes % 128 != 0 {
            bad.push(format!(
                "{}: {} B over 128 experts is not whole",
                d.name, d.nbytes
            ));
        }
        *per_expert.entry(l).or_default() += d.nbytes / 128;
        if part < 2 {
            types.entry(l).or_default()[part] = Some(ty);
        }
    }
    let mixed: BTreeSet<usize> = types
        .iter()
        .filter(|(_, t)| t[0] != t[1])
        .map(|(&l, _)| l)
        .collect();
    if mixed.len() != host_only {
        bad.push(format!(
            "{} layers of two gate·up types, wanted {host_only}",
            mixed.len()
        ));
    }

    let levers = PlanLevers::default();
    let batch = tier_batch(&inputs).unwrap_or_else(|e| panic!("{tag}: tier batch: {e}"));
    let machines = [
        ("a", workstation::plan_a(72)),
        ("gate", workstation::plan_gate(72)),
        ("bp", workstation::plan_bp(72, None, batch)),
    ];
    for (label, machine) in &machines {
        let plan = inputs
            .plan(machine, PLAN_CTX, &levers)
            .unwrap_or_else(|e| panic!("{tag}: plan {label}: {e}"));

        // The pool from the header, at a few pinned counts.
        for pinned in [0u64, 1, 3] {
            let kept = |l: usize| plan.n_l[l].saturating_sub(pinned);
            let want_bytes: u64 = (0..72)
                .map(|l| kept(l) * per_expert.get(&l).copied().unwrap_or(0))
                .sum();
            let want_experts: u64 = (0..72).map(kept).sum();
            let pool = ChurnPool::of(&plan, 0, pinned as usize)
                .unwrap_or_else(|e| panic!("{tag} {label}: pool at {pinned}: {e}"));
            if (pool.bytes, pool.experts) != (want_bytes, want_experts) {
                bad.push(format!(
                    "{tag} {label}: the pool pinning {pinned} holds {} experts, {} B; the header \
                     and the plan's card experts derive {want_experts}, {want_bytes} B",
                    pool.experts, pool.bytes
                ));
            }
        }

        // The layers that hold a pool run are the layers that hold card experts, and none is a
        // layer of two gate·up types.
        let pool = ChurnPool::of(&plan, 0, 0).unwrap_or_else(|e| panic!("{tag} {label}: {e}"));
        let pooled: BTreeSet<usize> = pool
            .runs
            .iter()
            .filter_map(|(t, _)| plan.model.tensors[*t].layer)
            .collect();
        let carded: BTreeSet<usize> = (0..72).filter(|&l| plan.n_l[l] > 0).collect();
        if pooled != carded || pooled.intersection(&mixed).next().is_some() {
            bad.push(format!(
                "{tag} {label}: the pool runs over layers {pooled:?}, the plan's card layers are \
                 {carded:?}, of two gate·up types {mixed:?}"
            ));
        }

        // The unset word. The gate placement is fixed; the serving placements run the serving
        // word as it is where the plan has room for it.
        let at = ResidencyAt {
            serving_place: *label != "gate",
            check_finite: false,
            route_trace: false,
            prefill_steps: false,
        };
        let pick = residency_at_plan(
            UnhostedPool::NotCarried,
            residency_unset(at),
            plan.n_l.iter().copied(),
            plan.host.experts,
            |p| ChurnPool::of(&plan, 0, p).map(|c| c.bytes),
            plan.host.headroom_bytes,
            plan.host.headroom_bytes,
        )
        .unwrap_or_else(|e| panic!("{tag} {label}: the unset word's pool: {e}"));
        let (word, why) = if *label == "gate" {
            ("off", ResidencyWhy::FixedPlace)
        } else {
            ("mid-p0-s1", ResidencyWhy::Place)
        };
        if (pick.word, pick.why) != (word, why) {
            bad.push(format!(
                "{tag} {label}: unset resolves to {} ({}), wanted {word} ({})",
                pick.word,
                pick.why.name(),
                why.name()
            ));
        }
        let steps = residency_unset(ResidencyAt {
            prefill_steps: true,
            ..at
        });
        if (steps.word, steps.why) != ("off", ResidencyWhy::PrefillSteps) && *label != "gate" {
            bad.push(format!(
                "{tag} {label}: the step feed resolves to {} ({})",
                steps.word,
                steps.why.name()
            ));
        }

        // The machine's start: a layer's card slots hold the word's pinned experts, its spare and
        // the expert that moves. Printed on the gate plan, whose default is `off`.
        let fewest = plan.n_l.iter().copied().filter(|&n| n > 0).min();
        if *label != "gate" && fewest.is_none_or(|n| n < 2) {
            bad.push(format!(
                "{tag} {label}: the fewest card experts a layer {fewest:?} hold no spare and an \
                 expert that moves"
            ));
        }
        let left = pool.check(&plan).map_err(|e| e.to_string());
        if *label != "gate" && left.is_err() {
            bad.push(format!(
                "{tag} {label}: the pool at the serving word: {left:?}"
            ));
        }
        let total: u64 = (0..72)
            .map(|l| per_expert.get(&l).copied().unwrap_or(0) * 128)
            .sum();
        println!(
            "{tag} residency {label} @ ctx {PLAN_CTX}: pool at the serving word {} experts, {} B \
             of {total} B routed ({:.2} % of the picks served by the stage card at uniform routing), \
             fewest card experts a layer {fewest:?}, host headroom {} B, left after the pool \
             {left:?}, unset {} ({}), {} layers of two gate·up types hold no pool run",
            pool.experts,
            pool.bytes,
            100.0 * pool.bytes as f64 / total as f64,
            plan.host.headroom_bytes,
            pick.word,
            pick.why.name(),
            mixed.len(),
        );
    }
    bad
}

/// The UD-IQ1_S header's residency on `a`, the gate plan and plan (b′): the churn pool at the
/// serving word `mid-p0-s1` is every card expert of the stage (4.48 GB on the gate plan, 30.04 GB
/// on plan (b′)'s stage, whose tier card's 23.32 GB the machine never moves [derived from the plan
/// tests above]), 2.09 % and 14.04 % of the 213.99 GB of routed bytes, which is the share of the
/// picks the stage card serves at uniform routing [derived]. The pool, the layers it runs over, the
/// unset word on each placement and the pool's fit in the plan's host headroom are the header's
/// and the plan's. Mutant: `ChurnPool::of` counting from rank `pinned + 1`, or the serving word
/// pinning one expert.
#[test]
fn mistrallarge4_residency_plan_ud_iq1_s() {
    let bad = residency_pins("ud-resid", KV, TENSORS, 0);
    assert!(
        bad.is_empty(),
        "{} pin(s) differ:\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}

/// The Mix-256G header's residency, the same plans: the 33 layers of two gate·up types hold no
/// card expert and so no pool run, the pool is the 38 other layers' stage experts (11.41 GB on the
/// gate plan, 36.99 GB on plan (b′)'s stage, 5.16 % and 16.72 % of the 221.28 GB of routed bytes
/// [derived from the plan tests above]).
#[test]
fn mistrallarge4_residency_plan_mix256() {
    let bad = residency_pins("mix-resid", MIX_KV, MIX_TENSORS, 33);
    assert!(
        bad.is_empty(),
        "{} pin(s) differ:\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}
