//! muse-spark metadata gate: what the header reader resolves from the Muse
//! Spark Open 816B-A42B UD-IQ2_XXS file's first shard
//! (unsloth/Muse-Spark-Open-816B-A42B-GGUF). The header test prints the
//! summary, then fails with every pin that differs. Headers only: the reader
//! holds every tensor to the dims the keys give it, so the tensor headers
//! are read and the bytes are not.
//!
//! `hw_`: needs the shards on the box (`BLOOMERY_REF_MODEL` names the first
//! shard; the box's path is the default).

use std::fs;

use models::{RopeMode, ToolFormat};

/// The box's path of the first shard.
const FILE: &str = "/data/moved/models/Muse-Spark-Open-816B-A42B-UD-IQ2_XXS/Muse-Spark-Open-816B-A42B-UD-IQ2_XXS-00001-of-00006.gguf";

/// Its length, so a different file at the same path fails before the pins.
const BYTES: u64 = 13_079_896;

/// The file's embedded chat template, in bytes (the specimen's own contract,
/// which differs from llama.cpp's `models/templates/muse-spark.jinja`).
const TEMPLATE: usize = 16_295;

/// The full layers (`sliding_window_pattern` false).
const FULL: &[usize] = &[1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61];

/// The first shard's path: the box's, or the one `BLOOMERY_REF_MODEL` names.
fn file() -> String {
    std::env::var_os("BLOOMERY_REF_MODEL").map_or_else(
        || FILE.to_string(),
        |v| v.into_string().expect("BLOOMERY_REF_MODEL must be UTF-8"),
    )
}

#[test]
#[ignore = "needs the Muse Spark Open shards on the box (BLOOMERY_REF_MODEL names the first)"]
fn hw_musespark_header() {
    let file = file();
    let mut bad = Vec::new();
    let len = fs::metadata(&file).map(|m| m.len());
    let split = gguf::Split::open(&file).unwrap_or_else(|e| panic!("open {file}: {e}"));
    let read = model::arch::spec(&split).unwrap_or_else(|e| panic!("spec of {file}: {e}"));
    let hp = model::arch::musespark::hparams::Hparams::read(&split)
        .unwrap_or_else(|e| panic!("hparams of {file}: {e}"));
    let unread = model::arch::musespark::hparams::unread_keys(&split);
    let s = &read.spec;
    let items: Vec<String> = model::arch::coverage::check(s, &read.tensors)
        .into_iter()
        .map(|u| u.feature)
        .collect();
    println!(
        "muse-spark file={file} bytes={len:?} kvs={} tensors={} layers={} sliding={} full={} \
         hidden={} vocab={} ctx={} window={} template={} defaults={:?} unread={unread:?}",
        split.shard(0).expect("the first shard").kv_count(),
        split.tensor_count(),
        s.layers.len(),
        hp.sliding.iter().filter(|&&s| s).count(),
        FULL.len(),
        s.hidden,
        s.vocab,
        s.ctx_train,
        hp.window,
        s.chat.template.as_ref().map_or(0, |t| t.len()),
        read.defaults,
    );
    println!("coverage: {items:?}");

    if len.as_ref().ok() != Some(&BYTES) {
        bad.push(format!("file bytes {len:?}, want {BYTES}"));
    }
    if s.arch != models::Arch::MuseSpark {
        bad.push(format!("arch {:?}", s.arch));
    }
    if (s.hidden, s.vocab, s.ctx_train) != (8192, 202_048, 1_048_576) {
        bad.push(format!(
            "hidden {} vocab {} ctx {}",
            s.hidden, s.vocab, s.ctx_train
        ));
    }
    if (s.rms_eps.to_bits(), hp.post_eps.to_bits()) != (0x3727_c5ac, 0x322b_cc77) {
        bad.push(format!(
            "rms_eps bits {:#x} post_eps bits {:#x}",
            s.rms_eps.to_bits(),
            hp.post_eps.to_bits()
        ));
    }
    if (hp.logit_scale.to_bits(), hp.softcap, hp.rope_base) != (0x3e35_04f3, 20.0, 500_000.0) {
        bad.push(format!(
            "logit_scale bits {:#x} softcap {} rope_base {}",
            hp.logit_scale.to_bits(),
            hp.softcap,
            hp.rope_base
        ));
    }
    if s.chat.pre != "gpt-4o"
        || s.chat.tools != Some(ToolFormat::MuseSpark)
        || s.chat.reasoning.is_some()
        || s.chat.template.as_ref().is_none_or(|t| t.len() != TEMPLATE)
    {
        bad.push(format!(
            "chat pre {} tools {:?} reasoning {:?} template {:?}",
            s.chat.pre,
            s.chat.tools,
            s.chat.reasoning,
            s.chat.template.as_ref().map(|t| t.len())
        ));
    }
    if read.defaults != ["rope.dimension_count = 64 (the head, llama-hparams.cpp)"] {
        bad.push(format!("defaults {:?}", read.defaults));
    }
    let want_unread = [
        "general.sampling.top_k",
        "general.sampling.top_p",
        "general.sampling.temp",
        "muse-spark.feed_forward_length",
        "muse-spark.rope.freq_base",
    ];
    if unread != want_unread {
        bad.push(format!("unread keys {unread:?}"));
    }
    if (s.layers.len(), hp.n_layer, hp.n_expert, hp.n_used) != (62, 62, 256, 8) {
        bad.push(format!(
            "{} layers, expert_count {} used {}",
            s.layers.len(),
            hp.n_expert,
            hp.n_used
        ));
    }
    for (l, layer) in s.layers.iter().enumerate() {
        let full = FULL.contains(&l);
        let models::Mixer::Gqa(g) = &layer.mixer else {
            bad.push(format!("layer {l}: {:?}", layer.mixer));
            continue;
        };
        if (g.heads, g.kv_heads, g.head_dim, g.value_dim) != (128, 16, 64, 64)
            || !g.qk_norm
            || g.out_gate != models::OutGate::Proj
            || g.select.is_some()
            || g.sinks
            || g.value_scale.is_some()
        {
            bad.push(format!("layer {l} gqa {g:?}"));
        }
        if full {
            if (g.rope.mode, g.rope.dims, g.rope.base) != (RopeMode::None, 0, 0.0)
                || g.window.is_some()
            {
                bad.push(format!("layer {l} full {:?} window {:?}", g.rope, g.window));
            }
        } else if (g.rope.mode, g.rope.dims, g.rope.base) != (RopeMode::NormTail, 64, 500_000.0)
            || g.window != Some(2048)
        {
            bad.push(format!(
                "layer {l} sliding {:?} window {:?}",
                g.rope, g.window
            ));
        }
        let models::Ffn::Moe(m) = &layer.ffn else {
            bad.push(format!("layer {l}: {:?}", layer.ffn));
            continue;
        };
        if (m.experts, m.top_k, m.expert_ff, m.latent) != (256, 8, 4096, Some(4096))
            || m.act != (models::Act::SwiGlu { limit: None })
            || m.router.score != models::Score::Sigmoid
            || !m.router.bias
            || m.router.norm != models::RouterNorm::SelectedSum
            || m.router.hash
            || m.router.scale != 1.0
            || m.shared.is_some()
        {
            bad.push(format!("layer {l} moe {m:?}"));
        }
    }
    let t = &read.tensors;
    if (t.layers, t.experts, t.experts_used) != (62, 256, 8) {
        bad.push(format!(
            "tensors' counts {} {} {}",
            t.layers, t.experts, t.experts_used
        ));
    }
    if t.tensors.len() != 1429 {
        bad.push(format!("tensors {}", t.tensors.len()));
    }
    let tensor = |name: &str| {
        t.tensors
            .iter()
            .find(|x| x.name == name)
            .map(|x| (x.role, x.ty, x.dims.clone()))
    };
    for (name, want) in [
        (
            "blk.0.attn_resid_alpha.weight",
            (models::Role::Attention, gguf::GgmlType::F32, vec![8192]),
        ),
        (
            "blk.0.ffn_latent_down.weight",
            (models::Role::Router, gguf::GgmlType::Q8_0, vec![8192, 4096]),
        ),
        (
            "blk.0.post_ffw_norm.weight",
            (models::Role::FfnNorm, gguf::GgmlType::F32, vec![8192]),
        ),
        (
            "blk.0.ffn_gate_exps.weight",
            (
                models::Role::RoutedExperts,
                gguf::GgmlType::IQ2_XXS,
                vec![4096, 4096, 256],
            ),
        ),
        (
            "blk.0.exp_probs_b.bias",
            (models::Role::Router, gguf::GgmlType::F32, vec![256]),
        ),
        (
            "token_embd.weight",
            (
                models::Role::TokenEmbedding,
                gguf::GgmlType::Q4_K,
                vec![8192, 202_048],
            ),
        ),
        (
            "output.weight",
            (
                models::Role::Head,
                gguf::GgmlType::Q6_K,
                vec![8192, 202_048],
            ),
        ),
    ] {
        if tensor(name) != Some(want) {
            bad.push(format!("{name} {:?}", tensor(name)));
        }
    }
    // The body runs the file: no feature of it is left to a program that does
    // not exist, which is what a placement of it needs (`PlanInputs::read`
    // refuses a file with any, every one listed).
    if !items.is_empty() {
        bad.push(format!("the coverage list is not empty: {items:?}"));
    }
    println!(
        "muse-spark layers=62 sliding={} full={} tensors=1429 template={TEMPLATE} \
         pins_differ={}",
        62 - FULL.len(),
        FULL.len(),
        bad.len()
    );
    assert!(
        bad.is_empty(),
        "{} pin(s) differ:\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}

/// The context the plan is read at: the release's.
const PLAN_CTX: u64 = 32_768;

#[test]
#[ignore = "needs the Muse Spark Open shards on the box (BLOOMERY_REF_MODEL names the first)"]
fn hw_musespark_plan() {
    use model::arch::musespark::host::post_exp_gain;
    use model::arch::musespark::hparams::Hparams;
    use model::arch::musespark::place::{self, PlanInputs};
    use model::placement::{PlanLevers, accepts, workstation};

    let file = file();
    let split = gguf::Split::open(&file).unwrap_or_else(|e| panic!("open {file}: {e}"));
    let hp = Hparams::read(&split).unwrap_or_else(|e| panic!("hparams of {file}: {e}"));
    let inputs =
        PlanInputs::read(&split).unwrap_or_else(|e| panic!("the plan's inputs of {file}: {e}"));
    let mut bad = Vec::new();
    // What the host leg reads beside the stacks: every layer's post-expert gain,
    // F32 of the latent's width, finite.
    for l in 0..hp.n_layer {
        match post_exp_gain(&split, &hp, l) {
            Ok(g) if g.len() == hp.latent && g.iter().all(|v| v.is_finite()) => {}
            other => bad.push(format!("layer {l} post-expert gain: {other:?}")),
        }
    }
    let total = (hp.n_layer * hp.n_expert) as u64;
    // The stage card of the release's placement (design §5 (a)): the A6000
    // runs every layer and the head, the rest of the experts on the host.
    let machine = workstation::placed_on(workstation::A6000, hp.n_layer, workstation::SCRATCH);
    let plan = inputs
        .plan(&machine, PLAN_CTX, &PlanLevers::default())
        .unwrap_or_else(|e| panic!("the plan on the A6000 at ctx {PLAN_CTX}: {e}"));
    let card = &plan.cards[0];
    let on_card: u64 = plan.n_l.iter().sum();
    println!(
        "muse-spark plan ctx={PLAN_CTX} A6000: dense {} experts {} ({} B) rounding {} kv {} \
         stage_rows {} scratch {} context {} headroom {}; host experts {} ({} B), nvme {} B, \
         room clause {}",
        card.dense_bytes,
        card.experts,
        card.expert_bytes,
        card.rounding_bytes,
        card.kv_bytes,
        card.stage_rows_bytes,
        card.scratch_bytes,
        card.context_bytes,
        card.headroom_bytes,
        plan.host.experts,
        plan.host.expert_bytes,
        plan.host.nvme_expert_bytes,
        plan.host.room_clause(),
    );
    if on_card + plan.host.experts != total || card.experts != on_card {
        bad.push(format!(
            "experts: {on_card} on the card (totals {}) + {} on the host, the file's {total}",
            card.experts, plan.host.experts
        ));
    }
    // The stack bytes where the plan puts each layer's experts, from the
    // headers: the card prefix in the card format the plan loads it in
    // (`accepts::leg_card_format`, measured by `CardFormat::buffer_bytes`: a
    // row of a whole number of words, so a stack whose rows end inside a word
    // takes more on the card than in the file), the rest in the file's bytes.
    let (mut card_stacks, mut host_stacks) = (0u64, 0u64);
    for t in inputs
        .model
        .tensors
        .iter()
        .filter(|t| t.role == models::Role::RoutedExperts)
    {
        let (Some(l), &[k, rows, experts]) = (t.layer, t.dims.as_slice()) else {
            panic!("{}: a routed stack of dims {:?}", t.name, t.dims);
        };
        let on = plan.n_l[l];
        let per_card: u64 = accepts::leg_card_format(t.ty)
            .buffer_bytes(t.ty, k, rows)
            .unwrap_or_else(|| panic!("{}: no card buffers of {rows} rows of {k}", t.name))
            .iter()
            .sum();
        card_stacks += on * per_card;
        host_stacks += (experts - on) * (t.file_bytes / experts);
    }
    if (card.expert_bytes, plan.host.expert_bytes) != (card_stacks, host_stacks) {
        bad.push(format!(
            "expert bytes: {} on the card + {} on the host, the stacks' {card_stacks} in the \
             card format + {host_stacks} in the file",
            card.expert_bytes, plan.host.expert_bytes
        ));
    }
    if plan.host.nvme_expert_bytes != 0 || plan.tier_n_l.iter().flatten().any(|&n| n != 0) {
        bad.push(format!(
            "paged or tiered experts: nvme {} B, tier counts {:?}",
            plan.host.nvme_expert_bytes, plan.tier_n_l
        ));
    }
    // The stores the body keeps: a plane of every position on every layer.
    let planes = inputs.kv.bytes(0..hp.n_layer, PLAN_CTX);
    if card.kv_bytes != planes {
        bad.push(format!(
            "kv bytes {} B, the planes' {planes} B",
            card.kv_bytes
        ));
    }
    // Plan (b′), the A6000 the stage and the 3090 its expert tier (`place::tier_batch` the tier's
    // prompt-batch bytes, the stage card held as the placed load holds it): the tier takes the ids
    // after the stage card's in each layer's stacks, the three places hold every expert once, and
    // the tier card's bytes are those stacks in the card format.
    let tiered = {
        let mut m = workstation::plan_tiers(
            hp.n_layer,
            workstation::A6000,
            &[workstation::RTX_3090],
            None,
            place::tier_batch(&hp).expect("valid tier geometry"),
        )
        .unwrap_or_else(|e| panic!("the tiered machine: {e}"));
        place::stage_placed(&mut m).unwrap_or_else(|e| panic!("the tiered machine's stage: {e}"));
        m
    };
    let tplan = inputs
        .plan(&tiered, PLAN_CTX, &PlanLevers::default())
        .unwrap_or_else(|e| panic!("the plan on the A6000 and the 3090 at ctx {PLAN_CTX}: {e}"));
    let on_stage: u64 = tplan.n_l.iter().sum();
    let on_tier: u64 = tplan.tier_n_l.iter().flatten().sum();
    if tplan.tier_n_l.len() != 1 || on_tier == 0 || on_stage + on_tier + tplan.host.experts != total
    {
        bad.push(format!(
            "tiered experts: {on_stage} on the stage card + {on_tier} on the tier ({} tier \
             cards' counts) + {} on the host, the file's {total}",
            tplan.tier_n_l.len(),
            tplan.host.experts
        ));
    }
    let mut tier_stacks = 0u64;
    for t in inputs
        .model
        .tensors
        .iter()
        .filter(|t| t.role == models::Role::RoutedExperts)
    {
        let (Some(l), &[k, rows, _]) = (t.layer, t.dims.as_slice()) else {
            panic!("{}: a routed stack of dims {:?}", t.name, t.dims);
        };
        let per_card: u64 = accepts::leg_card_format(t.ty)
            .buffer_bytes(t.ty, k, rows)
            .unwrap_or_else(|| panic!("{}: no card buffers of {rows} rows of {k}", t.name))
            .iter()
            .sum();
        tier_stacks += tplan.tier_n_l.first().map_or(0, |n| n[l]) * per_card;
    }
    let tier_card = tplan.cards.get(1).map_or(0, |c| c.expert_bytes);
    if tier_card != tier_stacks {
        bad.push(format!(
            "tier expert bytes: {tier_card} B on the tier card, the stacks' {tier_stacks} B in the \
             card format"
        ));
    }
    assert!(
        bad.is_empty(),
        "{} pin(s) differ:\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}
