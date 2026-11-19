//! hy_v4_flash metadata gate: what the header reader resolves from Tencent's Hy4-flash Q4_K_M file
//! (two shards), whose header tables are checked in beside this test (`hy4_hdr.kv.json`: the keys
//! of the first shard; `hy4_tensors.tsv`: every tensor of both shards, the 1,416 of the file).
//! The test builds the header alone as a GGUF (`common/header_tables.rs`: shapes, types and keys,
//! no tensor's bytes) and reads it through the engine's own `Split`; the reader holds every tensor
//! to the dims the keys give it and to the quant types the Q4_K_M set holds its stem in, so the
//! headers are the whole file as far as it reaches. One contract, one test: each prints its
//! summary, then fails with every pin that differs.
//!
//! The file's token map (`engram.token_map`, 120,832 values) is not checked in: the table holds its
//! summary, and [`token_map`] gives the header a map of the file's measured shape (the pad's
//! compressed id, the identity tail past the compressed range). The hash clause compares the
//! engine's rows with a transcription of llama.cpp's `engram_hash` on that map and the file's real
//! `ple.*` constants.
//!
//! The hy4 body runs the architecture ([`Program::Hy4Body`](model::placement::accepts::Program)):
//! the coverage clause holds every need the header's description names to a row of the table, so
//! the file's plan inputs read ([`hy4_body_covers_the_file`]).

use engram::Hash;
use gguf::{GgmlType, Split, Value};
use model::arch::hy4::hparams::{Hparams, unread_keys};
use model::arch::hy4::place::{CACHE_ROWS, PlanInputs};
use model::placement::workstation::{self, CardSpec};
use model::placement::{PlanLevers, Role};
use models::{
    Act, Arch, Collapse, EngramSpec, Extra, Ffn, HcKind, HcSpec, LatentOut, LatentUp, Mixer, Moe,
    NgramRule, OutGate, PoolRule, Residual, Rope, RopeMode, Router, RouterNorm, Score, Selector,
    Shared, Source,
};

use runtime::stores::{Access, PosBuf};

#[path = "common/hy4_header.rs"]
mod hy4_header;
use hy4_header::{KV, PAD_MAPPED, TENSORS, VOCAB, header, header_tables, kv_with, token_map};

/// The layers 0-based that run sparse attention: 0, 1 and every fourth from 5 through 41.
fn dsa(l: usize) -> bool {
    l < 2 || (l >= 5 && (l - 5).is_multiple_of(4))
}

/// The engram sites 0-based (`ple.layers`).
const SITES: [usize; 4] = [1, 11, 21, 31];

/// The header with the tensor table edited by `edit` and the key table by `edit_kv`.
fn header_edited(
    tag: &str,
    edit_kv: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>),
    edit: impl Fn(&str) -> Option<String>,
) -> Split {
    let tensors: String = TENSORS
        .lines()
        .filter_map(|l| edit(l).map(|l| format!("{l}\n")))
        .collect();
    header_tables::header_of(tag, &kv_with(edit_kv), &tensors)
}

/// The refusal text of reading `split`, or the empty string for a file that reads.
fn refusal(split: &Split) -> String {
    match model::arch::spec(split) {
        Ok(_) => String::new(),
        Err(e) => e.to_string(),
    }
}

/// The hy4 body covers every need of the file's description: the coverage list
/// (`arch::coverage::check`, the family's rows among the table's) is empty, so the plan inputs read
/// (`place::PlanInputs::describe`, then `runnable`). FAIL-first: a row of
/// `hy4::spec::AVAILABLE` taken out leaves its need listed, and the clause is red.
#[test]
fn hy4_body_covers_the_file() {
    let split = header("hy4-cover");
    let read = model::arch::hy4::spec::read(&split).unwrap_or_else(|e| panic!("read: {e}"));
    let missing = model::arch::coverage::check(&read.spec, &read.tensors);
    println!("hy4 coverage: {} needs no program runs", missing.len());
    for m in &missing {
        println!("  {m:?}");
    }
    let inputs = model::arch::hy4::place::PlanInputs::describe(&split)
        .and_then(|i| i.runnable())
        .map(|_| ())
        .map_err(|e| e.to_string());
    println!("hy4 plan inputs: {inputs:?}");
    assert!(
        missing.is_empty() && inputs.is_ok(),
        "the hy4 body leaves {} needs unmet; the inputs read {inputs:?}",
        missing.len()
    );
}

#[test]
fn hy4_header() {
    let split = header("hy4-meta");
    let read = model::arch::spec(&split).unwrap_or_else(|e| panic!("spec: {e}"));
    let hp = Hparams::read(&split).unwrap_or_else(|e| panic!("hparams: {e}"));
    let mut unread = unread_keys(&split);
    unread.sort();
    let s = &read.spec;
    let mut bad = Vec::new();
    println!(
        "hy_v4_flash tensors={} blocks={} layers={} dsa={} window={} sites={:?} hidden={} vocab={} \
         ctx={} template={} unread={unread:?}",
        split.tensor_count(),
        hp.n_block,
        s.layers.len(),
        (0..s.layers.len()).filter(|&l| hp.dsa(l)).count(),
        (0..s.layers.len()).filter(|&l| !hp.dsa(l)).count(),
        (0..s.layers.len())
            .filter(|&l| hp.site(l))
            .collect::<Vec<_>>(),
        s.hidden,
        s.vocab,
        s.ctx_train,
        s.chat.template.as_ref().map_or(0, |t| t.len()),
    );

    if s.arch != Arch::Hy4 {
        bad.push(format!("arch {:?}", s.arch));
    }
    if split.tensor_count() != 1416 || (hp.n_block, hp.n_layer) != (43, 42) {
        bad.push(format!(
            "tensors {} blocks {} layers {}",
            split.tensor_count(),
            hp.n_block,
            hp.n_layer
        ));
    }
    if (s.hidden, s.vocab, s.ctx_train) != (3072, 120_832, 1_048_576) {
        bad.push(format!(
            "hidden {} vocab {} ctx {}",
            s.hidden, s.vocab, s.ctx_train
        ));
    }
    if s.rms_eps.to_bits() != 1e-5f32.to_bits() {
        bad.push(format!("rms_eps {}", s.rms_eps));
    }
    if !read.defaults.is_empty() {
        bad.push(format!("defaults {:?}", read.defaults));
    }
    // Every architecture key is read; the sampling defaults the chat surface does not carry are
    // listed, never refused.
    if unread != ["general.sampling.temp", "general.sampling.top_p"] {
        bad.push(format!("unread keys {unread:?}"));
    }
    if s.chat.pre != "hy_v4"
        || s.chat.tools != Some(models::ToolFormat::Hy4)
        || s.chat.reasoning
            != Some(models::ReasoningFormat::ThinkSpan {
                open: "<think:6124c78e>",
                close: "</think:6124c78e>",
            })
        || s.chat.template.as_ref().is_none_or(|t| t.len() != 17_263)
    {
        bad.push(format!(
            "chat pre {} tools {:?} reasoning {:?} template {:?}",
            s.chat.pre,
            s.chat.tools,
            s.chat.reasoning,
            s.chat.template.as_ref().map(String::len)
        ));
    }
    if s.layers.len() != 42
        || !s.mtp.is_empty()
        || s.encoder.is_some()
        || s.logit_epilogue.is_some()
    {
        bad.push(format!(
            "{} layers, mtp {} encoder {:?}",
            s.layers.len(),
            s.mtp.len(),
            s.encoder
        ));
    }
    let want_hc = HcSpec {
        streams: 4,
        kind: HcKind::Identity {
            eps: 1e-6,
            magnitude: 2.0,
            collapse: Collapse::Head,
        },
    };
    if s.hc != Some(want_hc) {
        bad.push(format!("hc {:?}", s.hc));
    }
    let want_engram = EngramSpec {
        heads: 8,
        max_ngram: 3,
        key_length: 96,
        rule: NgramRule::EngramMasked,
    };
    if s.engram != Some(want_engram) {
        bad.push(format!("engram {:?}", s.engram));
    }

    let rope = |base: f32| Rope {
        mode: RopeMode::NormTail,
        dims: 64,
        base,
        yarn: None,
        factors: false,
    };
    let plain = Act::SwiGlu { limit: None };
    let want_moe = Moe {
        experts: 512,
        top_k: 16,
        expert_ff: 1536,
        latent: Some(1536),
        pre_exp_norm: false,
        post_exp_norm: None,
        sum_norm: false,
        act: Act::SwiGluGateFirst { limit: 10.0 },
        router: Router {
            score: Score::Sigmoid,
            bias: true,
            // The kept sum floored at 2^-14: the file sets no `expert_weights_norm_eps`.
            norm: RouterNorm::SelectedSumFloor,
            scale: 3.998,
            hash: false,
        },
        shared: Some(Shared {
            ff: 1536,
            act: plain,
            sigmoid_gate: false,
        }),
    };
    for (l, layer) in s.layers.iter().enumerate() {
        let sparse = dsa(l);
        let Mixer::Latent(a) = &layer.mixer else {
            bad.push(format!("layer {l}: mixer {:?}", layer.mixer));
            continue;
        };
        let base = if sparse { 1e7 } else { 1e4 };
        let heads = if sparse { 32 } else { 64 };
        if (a.heads, a.q_lora, a.latent, a.pe) != (heads, 1024, 512, 64)
            || a.up != (LatentUp::Absorbed { qk: 192, v: 256 })
            || a.rope != Some(rope(base))
            || a.out_gate != OutGate::Proj
            || a.out != LatentOut::Plain
            || !a.sinks
            || a.q_head_norm
            || a.ssmax
            || a.cache != Source::Own(())
            || a.compress.is_some()
            || a.window != (!sparse).then_some(512)
        {
            bad.push(format!("layer {l}: latent {a:?}"));
        }
        let want_select = sparse.then(|| Selector::TokenPool {
            heads: 16,
            d: 128,
            top_k: 2048,
            pool: 4,
            rule: PoolRule::Gated {
                key_eps: 1e-5,
                rope: rope(base),
            },
        });
        if a.select != want_select {
            bad.push(format!("layer {l}: select {:?}", a.select));
        }
        let want_ffn = if l == 0 {
            Ffn::Dense {
                ff: 13_824,
                act: plain,
            }
        } else {
            Ffn::Moe(want_moe.clone())
        };
        if layer.ffn != want_ffn {
            bad.push(format!("layer {l}: ffn {:?}", layer.ffn));
        }
        if layer.residual != (Residual::HcSandwich { eps: 1e-5 }) {
            bad.push(format!("layer {l}: residual {:?}", layer.residual));
        }
        let extras: &[Extra] = if SITES.contains(&l) {
            &[Extra::Engram]
        } else {
            &[]
        };
        if layer.extras != extras {
            bad.push(format!("layer {l}: extras {:?}", layer.extras));
        }
    }
    if (0..42).filter(|&l| dsa(l)).count() != 12 {
        bad.push("the pattern is not 12 sparse layers".to_string());
    }

    // The roles: the next-token block's every tensor unused, the table one gather of 4 sites x 16
    // columns on the first site's layer, the head's fold on the head, and no tensor of a role left
    // to the loader's choice.
    let t = &read.tensors;
    let nextn: Vec<_> = t.tensors.iter().filter(|x| x.layer == Some(42)).collect();
    if nextn.len() != 36 || nextn.iter().any(|x| x.role != Role::Unused) {
        bad.push(format!(
            "next-token block: {} tensors, roles {:?}",
            nextn.len(),
            nextn.iter().map(|x| x.role).collect::<Vec<_>>()
        ));
    }
    match t
        .tensors
        .iter()
        .find(|x| x.name == "per_layer_token_embd.weight")
    {
        Some(table)
            if table.role == Role::EngramTable
                && table.layer == Some(1)
                && table.gathered_rows == Some(64)
                && table.ty == GgmlType::F8E4M3
                && !table.ty.reads_scale()
                && table.dims == [96, 660_006_416] => {}
        other => bad.push(format!("per-layer table {other:?}")),
    }
    for name in [
        "output_hc_fn.weight",
        "output_hc_base.weight",
        "output_hc_scale.weight",
    ] {
        if t.tensors.iter().find(|x| x.name == name).map(|x| x.role) != Some(Role::Head) {
            bad.push(format!("{name} is not on the head"));
        }
    }
    if (t.layers, t.experts, t.experts_used) != (42, 512, 16) {
        bad.push(format!(
            "counts {} layers {} experts {} used",
            t.layers, t.experts, t.experts_used
        ));
    }
    // The by-position stores a plan holds in rooms: a sparse-attention layer holds one f16 latent row
    // (512 and 64 values) a position, of which the indexer's 2,048 are attended; a window layer
    // holds a ring of 512 latent rows and no room; the next-token block holds neither. The pooled
    // index keys (128 f16 values every 4 positions, scanned whole) are a plain buffer the plan counts
    // at the cap, so they are no room's.
    //
    // PIN(2026-11-19): the declaration was the cached rows and the pooled keys; it is the cached rows
    // alone. Derivation: the plan charges every declared buffer at the floor, and a scanned buffer
    // can grow past it only through a residency machine's freed expert bytes, which the Hy4 body
    // does not hold (`SLOT_MAP_FIXED`), so declaring the keys would under-count 12 x 262,144 x 256 B
    // = 805,306,368 B on the card at the cap. FAIL-first: the keys added back to the declaration
    // turn `hy4_plan_counts_its_rooms_at_the_floor` red on its cap-minus-floor difference.
    let cache = PosBuf {
        what: CACHE_ROWS,
        row: 1152,
        per: 1,
        runs: 1,
        access: Access::Select { rows: 2048 },
    };
    for l in 0..43 {
        let sparse = l < 42 && dsa(l);
        let want: Vec<PosBuf> = if sparse { vec![cache] } else { Vec::new() };
        let ring = if l < 42 && !sparse { 512 * 1152 } else { 0 };
        if hp.pos_bufs(l) != want || hp.ring_bytes(l) != ring {
            bad.push(format!(
                "layer {l}: by-position buffers {:?}, ring {} bytes",
                hp.pos_bufs(l),
                hp.ring_bytes(l)
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "hy4 header pins differ:\n{}",
        bad.join("\n")
    );
}

/// What the plan of the file at a context makes of its by-position rooms on a 32 GiB-class card: the
/// kind of the card [assumed: the card's totals are the fixture's, not a reading]. Derived by hand
/// (12 sparse layers, 1,152 B a cached row, 8,192 positions of floor, 1,048,576 of context): the cached
/// rows are the only room, so the plan's card K/V falls by exactly 12 x (1,048,576 - 8,192) x 1,152 =
/// 14,382,268,416 B when the lever sets the floor, whatever else the plan holds; the pooled index keys
/// (805,306,368 B), the rings and the scores stay at the cap on both sides, so a scanned buffer never
/// reads at the floor. The floor frees card bytes for routed experts (more of them on the card) and
/// leaves the host's expert bytes smaller by as many; `spill` keeps the card terms the floor sets, and
/// the plan names the floor and the word it was made under. A floor with no room declared is refused
/// by name (`KvUnroomed`).
///
/// FAIL-first: the rooms left off the plan's inputs (`with_rooms` removed from `describe`) refuses the
/// floor by name and the clause is red at the first plan; the pooled keys declared beside the cached
/// rows make the cap-minus-floor difference 12 x (262,144 - 2,048) x 256 = 799,014,912 B larger and it
/// is red at the difference.
#[test]
fn hy4_plan_counts_its_rooms_at_the_floor() {
    use bloomery_levers::KvHost;
    const MIB: u64 = 1 << 20;
    const CTX: u64 = 1_048_576;
    const FLOOR: u64 = 8_192;
    const CARD: CardSpec = CardSpec {
        name: "RTX_5090",
        total_bytes: 32_607 * MIB,
        driver_reserve_bytes: 550 * MIB,
        device: None,
        free_bytes: None,
        held_by: None,
        integrated: false,
        margin_bytes: None,
        granule_bytes: workstation::GRANULE,
    };
    let split = header("hy4-plan-floor");
    let mut inputs = PlanInputs::describe(&split).expect("the real header describes");
    inputs.kv_host_page = Some(2 * MIB);
    // The workstation's host, not this machine's reading of its own.
    inputs.room = (workstation::HOST_USABLE, workstation::HostRead::Given);
    let machine = workstation::plan_on(CARD, inputs.model.layers);
    let made = |levers: &PlanLevers| {
        let plan = inputs
            .plan(&machine, CTX, levers)
            .unwrap_or_else(|e| panic!("the card plans {CTX} positions: {e}"));
        (
            plan.cards.iter().map(|c| c.kv_bytes).sum::<u64>(),
            plan.n_l.iter().sum::<u64>(),
            plan.kv_floor,
            plan.kv_host,
            plan.host.expert_bytes,
            plan.host.kv_host_bytes,
        )
    };
    let unset = made(&PlanLevers::default());
    let floor = made(&PlanLevers {
        kv_floor: Some(FLOOR),
        kv_host: KvHost::Spill,
        ..PlanLevers::default()
    });
    println!(
        "hy4 plan at {CTX} positions on {}: unset kv {} B, {} card experts, host experts {} B; \
         floor {FLOOR} spill kv {} B, {} card experts, host experts {} B, host kv pages {} B",
        CARD.name, unset.0, unset.1, unset.4, floor.0, floor.1, floor.4, floor.5
    );
    let freed = 12 * (CTX - FLOOR) * 1152;
    assert_eq!(unset.0 - floor.0, freed, "the cached rows are the one room");
    // The host-able pages above the floor's, 2 MiB each: a layer's cached rows are 576 pages at the
    // context and 5 at the floor (9,437,184 B is 4.5 pages), so a spill sends 12 x 571 pages to the
    // host; no floor and no spill send none.
    assert_eq!(floor.5, 12 * 571 * 2 * MIB);
    assert_eq!(unset.5, 0);
    assert_eq!((floor.2, floor.3), (Some(FLOOR), KvHost::Spill));
    assert_eq!((unset.2, unset.3), (None, KvHost::Off));
    assert!(
        floor.1 > unset.1 && floor.4 < unset.4,
        "the freed bytes buy card experts: {} -> {} on the card, host {} -> {} B",
        unset.1,
        floor.1,
        unset.4,
        floor.4
    );
}

/// A streaming load's inputs keep the stream's ring beside a pass's bytes
/// ([`PlanInputs::with_pass_reserve`]): the pass's bytes replace the prompt reserve, and the ring's
/// floor joins after them, so the plan keeps the card experts of the reserve that is their sum,
/// fewer than the pass alone leaves, and counts the ring as the card's ring term. Plain inputs take
/// no ring. Mutant: the ring set first and the pass's bytes after (the reserve is the pass's alone
/// and the floor is lost: the reserve and the experts differ).
#[test]
fn hy4_pass_inputs_keep_the_ring_after_the_pass_bytes() {
    use model::arch::hy4::place::stream_ring;
    const MIB: u64 = 1 << 20;
    const PASS: u64 = 600 * MIB;
    const CTX: u64 = 8_192;
    const CARD: CardSpec = CardSpec {
        name: "RTX_5090",
        total_bytes: 32_607 * MIB,
        driver_reserve_bytes: 550 * MIB,
        device: None,
        free_bytes: None,
        held_by: None,
        integrated: false,
        margin_bytes: None,
        granule_bytes: workstation::GRANULE,
    };
    let split = header("hy4-pass-ring");
    let mut plain = PlanInputs::describe(&split).expect("the real header describes");
    plain.room = (workstation::HOST_USABLE, workstation::HostRead::Given);
    let ring = stream_ring(&plain).expect("the ring");
    let floor = ring.bytes(ring.floor);
    let machine = workstation::plan_on(CARD, plain.model.layers);
    // The experts a layer keeps on the cards, the stage card's prompt and ring bytes.
    let made = |inputs: &PlanInputs| {
        let plan = inputs
            .plan(&machine, CTX, &PlanLevers::default())
            .unwrap_or_else(|e| panic!("the card plans {CTX} positions: {e}"));
        (
            plan.n_l.clone(),
            plan.cards[0].prompt_bytes,
            plan.cards[0].ring_bytes,
        )
    };
    let kept = plain.clone().streaming(ring).with_pass_reserve(PASS);
    assert_eq!(
        kept.prompt_reserve,
        PASS + floor,
        "the pass's bytes, the floor after"
    );
    assert_eq!(kept.stream_slot, ring.slot);
    assert!(kept.ring.is_some());
    let bare = plain.clone().with_pass_reserve(PASS);
    assert_eq!((bare.prompt_reserve, bare.stream_slot), (PASS, 0));
    assert!(bare.ring.is_none(), "plain inputs take no ring");
    let (held, sum, alone) = (
        made(&kept),
        made(&plain.with_prompt_reserve(PASS + floor)),
        made(&bare),
    );
    let experts = |n_l: &[u64]| n_l.iter().sum::<u64>();
    println!(
        "hy4 pass {PASS} B, ring floor {floor} B: card experts {} (the reserve as a sum {}), without the ring {}; ring term {} B",
        experts(&held.0),
        experts(&sum.0),
        experts(&alone.0),
        held.2
    );
    assert_eq!(held.0, sum.0, "the experts of the reserve that is the sum");
    assert!(
        experts(&held.0) < experts(&alone.0),
        "the ring takes experts' room beside the pass"
    );
    assert_eq!(held.1, PASS, "the pass's bytes alone are the prompt term");
    assert!(held.2 >= floor, "the ring term holds the floor");
}

/// The reader refuses, by the name of the key or the tensor, what it does not run: each edit of the
/// real header below is one clause, and the unedited header reads (so a refusal is the edit's).
/// Mutant: the clause's check removed from the reader turns its message empty.
#[test]
fn hy4_refusals_name_what_is_wrong() {
    assert_eq!(refusal(&header("hy4-refuse-base")), "");
    let same = |_: &mut serde_json::Map<String, serde_json::Value>| {};
    let set = |key: &'static str, value: serde_json::Value| {
        move |m: &mut serde_json::Map<String, serde_json::Value>| {
            m.insert(key.to_string(), value);
        }
    };
    let mut bad = Vec::new();
    let mut want = |tag: &str, got: String, needle: &str| {
        if !got.contains(needle) {
            bad.push(format!(
                "{tag}: expected a refusal naming {needle:?}, got {got:?}"
            ));
        }
    };
    // The window pattern and the indexer's types must be each other's complement.
    let mut pattern = serde_json::from_str::<serde_json::Value>(KV).expect("parses")
        ["hy_v4_flash.attention.sliding_window_pattern"]
        .clone();
    pattern[2] = serde_json::Value::Bool(false);
    want(
        "sliding_window_pattern",
        refusal(&header_edited(
            "hy4-refuse-pattern",
            set("hy_v4_flash.attention.sliding_window_pattern", pattern),
            |l| Some(l.to_string()),
        )),
        "attention.sliding_window_pattern",
    );
    // A key head count other than 1.
    let mut kv_heads = vec![serde_json::Value::from(1); 43];
    kv_heads[7] = serde_json::Value::from(2);
    want(
        "head_count_kv",
        refusal(&header_edited(
            "hy4-refuse-kv",
            set("hy_v4_flash.attention.head_count_kv", kv_heads.into()),
            |l| Some(l.to_string()),
        )),
        "attention.head_count_kv",
    );
    // The selection without the query's own pool.
    want(
        "kpool_select_tail",
        refusal(&header_edited(
            "hy4-refuse-tail",
            set(
                "hy_v4_flash.attention.indexer.kpool_select_tail",
                false.into(),
            ),
            |l| Some(l.to_string()),
        )),
        "attention.indexer.kpool_select_tail",
    );
    // The per-layer table in bloomery's own scaled F8 rows, which demand a `.scale` tensor.
    want(
        "table type",
        refusal(&header_edited("hy4-refuse-f8rows", same, |l| {
            Some(l.replace(
                "per_layer_token_embd.weight\tF8E4M3",
                "per_layer_token_embd.weight\tF8E4M3_ROWS",
            ))
        })),
        "per_layer_token_embd.weight",
    );
    // An indexer tensor on a window layer, and a hyper-connection tensor on the next-token block.
    want(
        "misplaced indexer",
        refusal(&header_edited("hy4-refuse-indexer", same, |l| {
            Some(l.to_string()).map(|l| {
                if l.starts_with("blk.2.attn_norm.weight") {
                    format!("{l}\nblk.2.indexer.proj.weight\tF32\t3072x16\t1")
                } else {
                    l
                }
            })
        })),
        "blk.2.indexer.proj.weight",
    );
    want(
        "hyper-connection on the next-token block",
        refusal(&header_edited("hy4-refuse-hc42", same, |l| {
            Some(if l.starts_with("blk.42.attn_norm.weight") {
                format!("{l}\nblk.42.hc_attn_fn.weight\tF32\t12288x8\t1")
            } else {
                l.to_string()
            })
        })),
        "blk.42.hc_attn_fn.weight",
    );
    // A missing head fold, a tensor of the wrong dims, and a type outside the stem's list.
    want(
        "missing head fold",
        refusal(&header_edited("hy4-refuse-nohead", same, |l| {
            (!l.starts_with("output_hc_fn.weight")).then(|| l.to_string())
        })),
        "output_hc_fn.weight",
    );
    want(
        "dims",
        refusal(&header_edited("hy4-refuse-dims", same, |l| {
            Some(l.replace(
                "blk.3.attn_sinks.weight\tF32\t64",
                "blk.3.attn_sinks.weight\tF32\t32",
            ))
        })),
        "blk.3.attn_sinks.weight",
    );
    want(
        "type",
        refusal(&header_edited("hy4-refuse-type", same, |l| {
            Some(l.replace(
                "blk.3.ffn_gate_exps.weight\tQ4_K",
                "blk.3.ffn_gate_exps.weight\tQ6_K",
            ))
        })),
        "blk.3.ffn_gate_exps.weight",
    );
    // The map of another length, a pad outside it, and an unordered site list.
    want(
        "token map length",
        refusal(&header_edited(
            "hy4-refuse-map",
            |m| {
                m.insert(
                    "hy_v4_flash.engram.token_map".to_string(),
                    serde_json::Value::Array(vec![0.into(); VOCAB - 1]),
                );
            },
            |l| Some(l.to_string()),
        )),
        "engram.token_map",
    );
    want(
        "pad id",
        refusal(&header_edited(
            "hy4-refuse-pad",
            set("hy_v4_flash.engram.pad_token_id", (VOCAB as u64).into()),
            |l| Some(l.to_string()),
        )),
        "engram.pad_token_id",
    );
    want(
        "sites",
        refusal(&header_edited(
            "hy4-refuse-sites",
            set("hy_v4_flash.ple.layers", vec![11u32, 1, 21, 31].into()),
            |l| Some(l.to_string()),
        )),
        "ple.layers",
    );
    assert!(bad.is_empty(), "hy4 refusals differ:\n{}", bad.join("\n"));
}

/// llama.cpp's `engram_hash` (hy-v4-flash.cpp:338-384) over the header's keys, the position of each
/// token of `seq` (`None` an embedding position): per site, the row ids of every token, and
/// whether its own slot is live. i64 with the dead sentinel -1 as the C++ has it.
fn llamacpp_rows(
    split: &Split,
    map: &[u64],
    seq: &[Option<u32>],
) -> (Vec<Vec<Vec<u32>>>, Vec<bool>) {
    let get = |key: &str| -> Vec<u64> {
        match split.value(&format!("hy_v4_flash.{key}")).expect("the key") {
            Value::Array(a) => a
                .iter()
                .map(|v| v.as_unsigned().expect("a count"))
                .collect(),
            v => vec![v.as_unsigned().expect("a count")],
        }
    };
    let n_gram = get("ple.ngram_size")[0] as usize;
    let n_head = get("ple.heads_per_ngram")[0] as usize;
    let (mult, primes, offs) = (
        get("ple.layer_multipliers"),
        get("ple.head_vocab_sizes"),
        get("ple.head_offsets"),
    );
    let sites = get("ple.layers").len();
    let n_cols = (n_gram - 1) * n_head;
    let pad = map[get("engram.pad_token_id")[0] as usize] as i64;
    let cid = |t: i64| -> i64 {
        if t < 0 || t >= map.len() as i64 {
            -1
        } else {
            map[t as usize] as i64
        }
    };
    let mut out = vec![vec![vec![0u32; n_cols]; seq.len()]; sites];
    let mut valid = vec![false; seq.len()];
    for (i, t) in seq.iter().enumerate() {
        let c = t.map_or(-1, |t| cid(i64::from(t)));
        valid[i] = c != -1;
        for e in 0..sites {
            let mut toks = vec![0u64; n_gram];
            toks[0] = (if c == -1 { pad } else { c }) as u64;
            for s in 1..n_gram {
                let c = if (i as i64) - (s as i64) < 0 {
                    -1
                } else {
                    seq[i - s].map_or(-1, |t| cid(i64::from(t)))
                };
                toks[s] = (if c == -1 { pad } else { c }) as u64;
            }
            let m = &mult[e * n_gram..][..n_gram];
            let mut rolling = toks[0].wrapping_mul(m[0]);
            for o in 1..n_gram {
                rolling ^= toks[o].wrapping_mul(m[o]);
                for h in 0..n_head {
                    let col = (o - 1) * n_head + h;
                    let (p, of) = (primes[e * n_cols + col], offs[e * n_cols + col]);
                    out[e][i][col] = u32::try_from(rolling % p + of).expect("a row inside u32");
                }
            }
        }
    }
    (out, valid)
}

/// The engine's engram hash, built from the header's own keys, gives the rows and the masks of the
/// transcription on fixed ids: a text prompt, the pad token, ids of the identity tail, an id past
/// the map (dead) and an embedding position (dead), at one call and in calls of 1, 3 and 5 tokens.
/// Every row id stays inside the table's row count, and the pad slot is the map's value at the pad.
/// Mutant: the pad slot hashed as the raw pad id, or a dead token's mask dropped.
#[test]
fn hy4_engram_rows_are_llamacpps() {
    let split = header("hy4-hash");
    let hash = Hash::ple_mapped_from_keys(split.keys(), "hy_v4_flash", VOCAB as u32)
        .unwrap_or_else(|e| panic!("hash: {e}"));
    let mut bad = Vec::new();
    if (
        hash.sites(),
        hash.n_gram(),
        hash.n_heads(),
        hash.n_cols(),
        hash.key_length(),
    ) != (4, 3, 8, 16, 96)
        || hash.layer_ids() != [1, 11, 21, 31]
    {
        bad.push(format!(
            "shape {} sites, {}-gram, {} heads, {} columns, rows of {}, layers {:?}",
            hash.sites(),
            hash.n_gram(),
            hash.n_heads(),
            hash.n_cols(),
            hash.key_length(),
            hash.layer_ids()
        ));
    }
    if hash.pad_id() != PAD_MAPPED {
        bad.push(format!(
            "pad slot {} (the map's value at the pad)",
            hash.pad_id()
        ));
    }
    let map = token_map();
    let ids: Vec<u32> = vec![
        5, 17, 90_001, 120_002, 120_817, 120_818, 120_831, 33, 120_832, 4_000_000, 2, 120_002, 99,
        61_000, 7, 7, 7, 120_825, 8, 31_415,
    ];
    let mut seq: Vec<Option<u32>> = ids.iter().map(|&t| Some(t)).collect();
    seq[10] = None;
    let (want, want_live) = llamacpp_rows(&split, &map, &seq);
    let rows_total = 660_006_416u32;
    for step in [seq.len(), 1, 3, 5] {
        let mut hist = hash.new_mapped_history().expect("a mapped history");
        let mut at = 0;
        while at < seq.len() {
            let chunk = &seq[at..(at + step).min(seq.len())];
            let mut rows = vec![0u32; hash.sites() * chunk.len() * hash.n_cols()];
            let mut live = vec![false; chunk.len()];
            hash.mapped_rows_into(&mut hist, at as u64, chunk, &mut rows, &mut live)
                .unwrap_or_else(|e| panic!("rows at {at}: {e}"));
            for (i, &l) in live.iter().enumerate() {
                if l != want_live[at + i] {
                    bad.push(format!("live[{}] {l} at step {step}", at + i));
                }
                for e in 0..hash.sites() {
                    let got = &rows[(e * chunk.len() + i) * hash.n_cols()..][..hash.n_cols()];
                    if got != &want[e][at + i][..] {
                        bad.push(format!("site {e} token {} at step {step}", at + i));
                    }
                    if got.iter().any(|&r| r >= rows_total) {
                        bad.push(format!("site {e} token {}: a row past the table", at + i));
                    }
                }
            }
            at += chunk.len();
        }
        if hist.next_pos() != seq.len() as u64 {
            bad.push(format!(
                "history at {} after {} tokens",
                hist.next_pos(),
                seq.len()
            ));
        }
    }
    assert!(bad.is_empty(), "hy4 hash pins differ:\n{}", bad.join("\n"));
}
