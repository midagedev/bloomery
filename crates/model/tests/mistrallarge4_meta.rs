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
//! No program runs the architecture yet, so the coverage check's list is
//! printed and its new parts pinned, not empty — as LFM3's gate holds its
//! own. The context bound (`spec::serve_ctx`) is pinned here too.

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
    let kvs: Vec<(String, Value)> = serde_json::from_str::<serde_json::Value>(KV)
        .expect("the checked-in header table parses")
        .as_object()
        .expect("the header table is an object")
        .into_iter()
        // The fetch summarized the four tokenizer arrays the reader never
        // reads (`{array_len, head}`); the token list's own key the synthetic
        // readers write is the reader's too, never read here.
        .filter_map(|(k, v)| value(k, v).map(|v| (k.clone(), v)))
        .collect();
    let tensors: Vec<TensorDecl> = TENSORS
        .lines()
        .map(|line| {
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
            let per_row =
                ty.type_size().expect("a size") * (dims[0] / ty.blck_size().expect("a block"));
            TensorDecl {
                name: name.to_string(),
                dims,
                type_id: ty.as_u32(),
                nbytes: per_row * rows,
            }
        })
        .collect();
    let layout = Layout::new(&kvs, tensors).expect("the header lays out");
    let path = std::env::temp_dir().join(format!("bloomery-ml4-meta-{}.gguf", std::process::id()));
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
    // No program runs the file: the coverage list names every part of it,
    // the new ones among them — pinned here, emptied by the body's round.
    // What no common row runs: the two latent shapes (the absorbed
    // query with and without the rope part), SSMax, the shared cache and the
    // keys from the layer input, beside the program and the tool parser.
    // The pre-tokenizer `tekken-num3` is a known one, so no item. The
    // unrotated-key row, the 32x128 stream indexer and the softmax router
    // shapes are common rows' (the V4.1 indexer and the shape table), so they
    // are no items here.
    for want in [
        "latent attention 512 x rope 64, absorbed qk 64 / v 128",
        "latent attention 512 x rope 0, absorbed qk 128 / v 128",
        "SSMax: the query scaled per head by s*ln(pos + 1) + b",
        "the latent cache another layer wrote, read in place of its own",
        "index keys projected from the layer input, one per position",
        "a layer program for mistral-large4",
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
    if got != (5, 1024, 16, 16, 256, 4096, 16) {
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
    // The reader serves every position the fixture trained (64), past the
    // indexer's own top-k 16 among them; past the trained one is named.
    // PIN(2026-11-01): R4a's card tier lifted the top-k bound (above).
    let ctx = model::arch::mistrallarge4::spec::serve_ctx;
    if ctx(16, s) != Ok(64) || ctx(64, s) != Ok(64) || ctx(65, s).is_ok() {
        bad.push(format!(
            "serve_ctx(16) = {:?}, (64) = {:?}, (65) = {:?}",
            ctx(16, s),
            ctx(64, s),
            ctx(65, s)
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
            (GgmlType::IQ2_XXS, vec![256, 1024, 16]),
        ),
        (
            "blk.2.ffn_gate_exps.weight",
            (GgmlType::IQ1_S, vec![1024, 256, 16]),
        ),
        (
            "blk.2.ffn_down_exps.weight",
            (GgmlType::IQ2_XXS, vec![256, 1024, 16]),
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
            (GgmlType::IQ1_S, vec![256, 1024, 16]),
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
