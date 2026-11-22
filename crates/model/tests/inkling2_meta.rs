//! inkling2 metadata gate: what the header reader `arch::inkling2` resolves from the two Inkling 2
//! files on the box — the Mini's looped `inkling2` (unsloth's UD-Q4_K_XL) and the Small's loopless
//! one (unsloth's UD-Q2_K_XL, six shards) — what the coverage check lists for them (the parts no
//! program in this tree runs yet, the work queue of the rounds that build one), the tensors by
//! role and type, the trunk's layer visits and cache slots (a visit a slot, the loop's core
//! repeated), and the placement's own print: the weights once per stored layer, the stores once a
//! visit at three depths, and the host-routed plan on the gate machine. One contract, one test a
//! file; each prints what it compared, then fails with the whole list of what differs.
//!
//! `hw_inkling2_fixture` is the fixture-tier clause: the family's planted fixture
//! (`arch::inkling2::fixture::spec`, thirteen layers of the Mini) read back through the same
//! reader, at its planted path (the file the fixture tier resolves).
//!
//! `hw_`: needs the files on the box (`just gate-inkling2-meta`). Headers only: seconds.

#[path = "common/spec_view.rs"]
mod spec_view;

use std::collections::BTreeMap;
use std::fmt::Write as _;

use gguf::Split;
use model::arch::inkling2::place;
use model::placement::{KvBytes, PlanLevers, Role, workstation};

/// The Mini: unsloth's UD-Q4_K_XL, one shard, the loop of the core [6, 18) three times.
const MINI: &str = "/models/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf";

/// The Small: unsloth's UD-Q2_K_XL, six shards, no loop.
const SMALL: &str =
    "/models/Inkling-2-Small-GGUF/UD-Q2_K_XL/Inkling-2-Small-UD-Q2_K_XL-00001-of-00006.gguf";

// PIN(2026-11-21): the Mini's description as `arch::inkling2::spec` reads it, line by line (the
// key values checked against the UD-Q4_K_XL header: 24 stored layers of 48 visits, heads 24/8 of
// 128, 128 experts of 1152 at the top 8 scaled 8, the bias bank 16 x 1024, the clips 22.627417
// and 8, the log-scaled positions at floor 8192 alpha 0.1, four taps, the windowed layers at
// 1024, the windowless at 5, 11, 17 and 23 and their repeats, the loop of [6, 18) three times at
// rank 64, the head softcapped at 30 and cut to the unpadded 200058 of 201024).
const MINI_VIEW: &[&str] = &[
    "arch Inkling2",
    "hidden 2304 vocab 201024 ctx_train 1048576",
    "rms_eps bits 0x358637bd",
    "layers 24 mtp 0",
    "hc None",
    "engram None",
    "chat pre inkling2 template bytes 6868 tools None reasoning None",
    "loop core [6, 18) passes 3 rank 64",
    "logit scale 1 softcap 30 cut 200058",
    "[0-4,6-10,12-16,18-22] relbias 24/8 x 128 d_rel 16 extent 1024 clips 22.627417/8 log 8192/0.1 taps 4 win 1024 gate Proj || moe 128/8 ff 1152 situ 4/25 Sigmoid bias true norm true x8 hash false || attn-res-mix 6/1",
    "[5,11,17,23] relbias 24/8 x 128 d_rel 16 extent 1024 clips 22.627417/8 log 8192/0.1 taps 4 dyn gate Proj || moe 128/8 ff 1152 situ 4/25 Sigmoid bias true norm true x8 hash false || attn-res-mix 6/3",
];

// PIN(2026-11-21): the Mini takes no key default; its `defaults` are the description row's facts,
// each with its source (the router's f32 logits and selected-sum norm, the value norm, the mix's
// fixed epsilon, the convolutions' scale, the sink denominator, the head's cut, flash attention
// off), in the row's order.
const MINI_DEFAULTS: &[&str] = &[
    "router norm = selected-sum (B:65: expert_weights_norm_eps = 1e-20 unconditional)",
    "router logits = f32 (expert_router_f32) (B:64: the router scores run in f32)",
    "router bias = selection only (M:690-696: exp_probs_b steers the choice, the kept weights are the unbiased scores)",
    "V = gainless per-head RMS, always (B:406: the value rows are normed whatever the file's `value_norm` key says)",
    "mix eps = 1e-6 fixed, scale 1/sqrt(hidden) (B:217,222: the block attention residual's states are normed at a fixed epsilon and scored at 1/sqrt(hidden))",
    "conv output = (x + conv(x)) / sqrt(taps + 1), twice the taps where the k/v convs are dynamic (B:282: every convolution's scale counts its taps (and the dynamic taps))",
    "softmax denominator = 1 + sum(exp(s - t)) (B:429-438: a per-(head, query) sink logit t adds a zero column)",
    "head = softcap then the unpadded columns (B:503-510: the softcap runs on the padded logits, the cut after it)",
    "flash attention = off (B:88, llama-graph.cpp:3201: use_flash_attn needs no kq_b, and every layer carries one)",
];

// PIN(2026-11-21): the keys of the Mini the reader does not read: the rope it never turns, the
// keys' second spelling the reference abandoned, the dense width no layer holds and the imatrix
// calibration's.
const MINI_UNREAD: &[&str] = &[
    "inkling2.feed_forward_length",
    "inkling2.attention.key_length_swa",
    "inkling2.attention.value_length_swa",
    "inkling2.rope.dimension_count",
    "general.sampling.temp",
    "general.sampling.top_k",
    "general.sampling.top_p",
    "general.sampling.min_p",
];

// PIN(2026-11-21): every tensor of the Mini by role and type (734 in all), [derived] from the
// header dump's per-name table: the attention rows Q8_0 but the sink projection BF16 and the
// small matrices F32; the loop's controllers and queries F32; the routed stacks NVFP4 on the
// first twenty layers and Q8_0 on the last four, their per-expert scales unread F32.
const MINI_TENSORS: &[&str] = &[
    "attn bf16 32",
    "attn f32 345",
    "attn q8_0 144",
    "ffn_norm f32 72",
    "head f32 2",
    "head q8_0 1",
    "routed nvfp4 40",
    "routed q8_0 8",
    "router f32 48",
    "token_embd f32 1",
    "token_embd q8_0 1",
    "unread f32 40",
];

// PIN(2026-11-21): the Small's description: the same program at twice the width, no loop, the
// GPT-4o spelling of the pre-tokenizer, 40 windowed layers of 48 and the windowless at every
// sixth from 5. The log-scaled positions run at the file's floor 128000 — `attention.log_scale_floor`,
// the one key the reference reads — where the checkpoint's config.json says 8192: the description
// carries the reference's value, the divergence recorded here.
const SMALL_VIEW: &[&str] = &[
    "arch Inkling2",
    "hidden 4608 vocab 201024 ctx_train 1048576",
    "rms_eps bits 0x358637bd",
    "layers 48 mtp 0",
    "hc None",
    "engram None",
    "chat pre gpt-4o template bytes 27995 tools None reasoning None",
    "logit scale 1 softcap 30 cut 200058",
    "[0-4,6-10,12-16,18-22,24-28,30-34,36-40,42-46] relbias 48/8 x 128 d_rel 16 extent 1024 clips 22.627417/8 log 128000/0.1 taps 4 win 1024 gate Proj || moe 384/8 ff 2304 situ 4/25 Sigmoid bias true norm true x8 hash false || attn-res-mix 6/1",
    "[5,11,17,23,29,35,41,47] relbias 48/8 x 128 d_rel 16 extent 1024 clips 22.627417/8 log 128000/0.1 taps 4 dyn gate Proj || moe 384/8 ff 2304 situ 4/25 Sigmoid bias true norm true x8 hash false || attn-res-mix 6/3",
];

const SMALL_DEFAULTS: &[&str] = &[
    "router norm = selected-sum (B:65: expert_weights_norm_eps = 1e-20 unconditional)",
    "router logits = f32 (expert_router_f32) (B:64: the router scores run in f32)",
    "router bias = selection only (M:690-696: exp_probs_b steers the choice, the kept weights are the unbiased scores)",
    "V = gainless per-head RMS, always (B:406: the value rows are normed whatever the file's `value_norm` key says)",
    "mix eps = 1e-6 fixed, scale 1/sqrt(hidden) (B:217,222: the block attention residual's states are normed at a fixed epsilon and scored at 1/sqrt(hidden))",
    "conv output = (x + conv(x)) / sqrt(taps + 1), twice the taps where the k/v convs are dynamic (B:282: every convolution's scale counts its taps (and the dynamic taps))",
    "softmax denominator = 1 + sum(exp(s - t)) (B:429-438: a per-(head, query) sink logit t adds a zero column)",
    "head = softcap then the unpadded columns (B:503-510: the softcap runs on the padded logits, the cut after it)",
    "flash attention = off (B:88, llama-graph.cpp:3201: use_flash_attn needs no kq_b, and every layer carries one)",
];

const SMALL_UNREAD: &[&str] = &[
    "inkling2.feed_forward_length",
    "inkling2.d_rel",
    "inkling2.rel_extent",
    "inkling2.rel_extent_swa",
    "inkling2.shortconv_kernel",
    "inkling2.log_scaling_n_floor",
    "inkling2.log_scaling_alpha",
    "inkling2.unpadded_vocab_size",
    "inkling2.qk_clip_threshold",
    "inkling2.rel_clip_threshold",
    "inkling2.attn_res_normalize_values",
    "inkling2.mlp_gate_softcap",
    "inkling2.mlp_up_softcap",
    "inkling2.value_norm",
];

const SMALL_TENSORS: &[&str] = &[
    "attn bf16 48",
    "attn f32 415",
    "attn q6_K 120",
    "attn q8_0 168",
    "ffn_norm f32 144",
    "head f32 2",
    "head q6_K 1",
    "routed iq2_xs 40",
    "routed iq3_xxs 48",
    "routed iq4_xs 8",
    "router f32 96",
    "token_embd f32 1",
    "token_embd q5_K 1",
];

/// The trunk's layer visits as pins: the count over the stored layers, the run order as the layer
/// runs of each segment (the reference's: the stem, the core's passes one after another, the
/// coda), and the slot table's windowless slots — a slot a visit, windowed by its stored layer.
fn visit_lines(spec: &models::ModelSpec) -> Vec<String> {
    let visits = spec.visits();
    let mut out = vec![format!(
        "visits {} over {} stored layers",
        visits.len(),
        spec.layers.len()
    )];
    let mut runs: Vec<String> = Vec::new();
    let mut start = 0;
    for i in 0..=visits.len() {
        let new_run = i == visits.len()
            || i > start
                && (visits[i].layer != visits[i - 1].layer + 1
                    || visits[i].pass != visits[i - 1].pass);
        if new_run {
            runs.push(format!(
                "{}-{} p{}",
                visits[start].layer,
                visits[i - 1].layer,
                visits[start].pass
            ));
            start = i;
        }
    }
    out.push(format!("runs {}", runs.join(" | ")));
    let windowless: Vec<String> = visits
        .iter()
        .enumerate()
        .filter(|(_, v)| {
            matches!(
                &spec.layers[v.layer as usize].mixer,
                models::Mixer::RelBias(a) if a.window.is_none()
            )
        })
        .map(|(slot, _)| slot.to_string())
        .collect();
    out.push(format!("windowless slots {}", windowless.join(",")));
    out
}

/// One file's description, defaults, unread keys, tensors, coverage items, layer visits and
/// placement: printed, each compared against its pins.
fn check(
    path: &str,
    view: &[&str],
    defaults: &[&str],
    unread_pins: &[&str],
    tensors: &[&str],
    visits: &[&str],
    plan: &[&str],
) {
    let mut o = String::new();
    let mut b = Vec::new();
    let split = Split::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let read = model::arch::spec(&split).unwrap_or_else(|e| panic!("spec of {path}: {e}"));
    let _ = writeln!(o, "file {path}");
    spec_view::compare(
        &mut o,
        &mut b,
        "description",
        &spec_view::view(&read.spec),
        view,
    );
    spec_view::compare(&mut o, &mut b, "defaults taken", &read.defaults, defaults);
    let unread = model::arch::inkling2::hparams::unread_keys(&split);
    spec_view::compare(&mut o, &mut b, "keys not read", &unread, unread_pins);
    let mut by_role: BTreeMap<(String, String), usize> = BTreeMap::new();
    for t in &read.tensors.tensors {
        *by_role
            .entry((t.role.to_string(), t.ty.to_string()))
            .or_default() += 1;
    }
    let got: Vec<String> = by_role
        .iter()
        .map(|((role, ty), n)| format!("{role} {ty} {n}"))
        .collect();
    let _ = writeln!(o, "tensors {}", read.tensors.tensors.len());
    spec_view::compare(&mut o, &mut b, "tensors by role and type", &got, tensors);
    spec_view::compare(
        &mut o,
        &mut b,
        "visits and slots",
        &visit_lines(&read.spec),
        visits,
    );
    let list = model::arch::coverage::check(&read.spec, &read.tensors);
    let _ = writeln!(o, "coverage (the work queue, printed not pinned)");
    for line in spec_view::items(&list) {
        let _ = writeln!(o, "  {line}");
    }
    spec_view::compare(
        &mut o,
        &mut b,
        "stores and plan",
        &place_lines(&split),
        plan,
    );
    let _ = o; // printed below
    println!("{o}");
    assert!(
        b.is_empty(),
        "{} pin(s) differ:\n  {}",
        b.len(),
        b.join("\n  ")
    );
}

/// The placement's own print of `split`: the weights once per stored layer (the routed stacks and
/// the rest), the trunk's stores once a visit at three depths and the conv state's share, and the
/// host-routed plan on the gate machine (no card kernel reads this family's experts yet, so every
/// stack sits on the host and the card holds the dense weights and the stores).
fn place_lines(split: &Split) -> Vec<String> {
    let inputs = place::Inputs::describe(split).expect("the file describes");
    let (mut dense, mut routed) = (0u64, 0u64);
    for t in &inputs.model.tensors {
        if t.role == Role::RoutedExperts {
            routed += t.file_bytes;
        } else if t.role != Role::Unread {
            dense += t.file_bytes;
        }
    }
    let kv = &inputs.kv;
    let at = |ctx: u64| {
        (0..inputs.model.layers)
            .map(|l| kv.layer_bytes(l, ctx))
            .sum::<u64>()
    };
    let mut out = vec![
        format!("weights dense {dense} routed {routed} B"),
        format!(
            "stores 6 {} B 4096 {} B 262144 {} B",
            at(6),
            at(4096),
            at(262_144)
        ),
    ];
    let machine = workstation::plan_gate(inputs.model.layers);
    let plan = inputs
        .plan(&machine, 4096, &PlanLevers::default())
        .expect("the gate machine plans the file");
    let stage = &plan.cards[0];
    out.push(format!(
        "plan@4096 dense {} kv {} scratch {} context {} | host experts {} B of {} experts",
        stage.dense_bytes,
        stage.kv_bytes,
        stage.scratch_bytes,
        stage.context_bytes,
        plan.host.expert_bytes,
        plan.host.experts,
    ));
    out
}

#[test]
#[ignore = "needs the Inkling-2-Mini UD-Q4_K_XL file on the box"]
fn hw_inkling2_spec_mini() {
    check(
        MINI,
        MINI_VIEW,
        MINI_DEFAULTS,
        MINI_UNREAD,
        MINI_TENSORS,
        MINI_VISITS,
        MINI_PLAN,
    );
}

#[test]
#[ignore = "needs the Inkling-2-Small UD-Q2_K_XL shards on the box"]
fn hw_inkling2_spec_small() {
    check(
        SMALL,
        SMALL_VIEW,
        SMALL_DEFAULTS,
        SMALL_UNREAD,
        SMALL_TENSORS,
        SMALL_VISITS,
        SMALL_PLAN,
    );
}

// PIN(2026-11-21): the Mini's layer visits and slot table [derived from the description the
// reader holds and the reference's run order (inkling2.cpp's graph loop)]: 48 visits over 24
// stored layers — the stem 0-5, the core 6-17 three times, the coda 18-23 — the windowless slots
// the windowless layers' (5 and 23 once, 11 and 17 three times each: 8 of the 48).
const MINI_VISITS: &[&str] = &[
    "visits 48 over 24 stored layers",
    "runs 0-17 p0 | 6-17 p1 | 6-17 p2 | 18-23 p0",
    "windowless slots 5,11,17,23,29,35,41,47",
];

// PIN(2026-11-21): the Small's: no loop, the visits the layers, the windowless slots every sixth
// from 5.
const SMALL_VISITS: &[&str] = &[
    "visits 48 over 48 stored layers",
    "runs 0-47 p0",
    "windowless slots 5,11,17,23,29,35,41,47",
];

// PIN(2026-11-21): the Mini's weights, stores and host-routed plan [derived, the terms the reader
// holds]: the dense weights (every tensor but the routed stacks and the unread scales) and the
// routed stacks of ink2r0 §2.1's 17,617,780,736 B; a visit's K and V of 8·2·128 f16 values
// (4,096 B) a position, its conv state (2·1,024 + 2·2,304)·3 f32 = 79,872 B, the 40 windowed
// visits' rows on the ring (`ring_rows(1,024, ·)` = the window and a prompt batch, aligned:
// 1,536 rows once filled — ink2r0 §4's KV table counts the bare 1,024 rows and the conv state
// apart, the two terms this line folds in). The plan line is the gate machine's
// (`workstation::plan_gate`, a 24 GB card, not the 5090 of ink2r0 §4): the card the dense weights,
// the stores and the machine's terms, the host every routed stack — the card-kernel round
// replaces this plan with the engine's own.
const MINI_PLAN: &[&str] = &[
    "weights dense 1819914240 routed 15797846016 B",
    "stores 6 5013504 B 4096 389709824 B 262144 8845426688 B",
    "plan@4096 dense 1331632128 kv 389709824 scratch 67108864 context 536870912 | host experts 15797846016 B of 3072 experts",
];

// PIN(2026-11-21): the Small's, the same terms at 8 key heads of 128 and hidden 4,608 (a conv
// state of 135,168 B a visit): ink2s0 §4.3's table is the bare-window arithmetic plus its "~4 MB"
// of conv states; this line is the ring's pad and the exact conv states folded in.
const SMALL_PLAN: &[&str] = &[
    "weights dense 5955132672 routed 199001899008 B",
    "stores 6 7667712 B 4096 392364032 B 262144 8848080896 B",
    "plan@4096 dense 5339503872 kv 392364032 scratch 67108864 context 536870912 | host experts 199001899008 B of 18432 experts",
];

/// The fixture's planted path under the fixture root (`arch::inkling2::fixture::spec`'s
/// directory): the fixture-tier file the gate reads there.
const FIXTURE: &str = "/models/fixtures/inkling2/inkling2-fixture-00001-of-00001.gguf";

/// The fixture's whole length, so a different file at the same path fails before the pins.
const FIXTURE_BYTES: u64 = 4_983_014_240;

// PIN(2026-11-21): the fixture's identity and description [derived: the generator's bytes are a
// function of the seed and the source header alone (`model::fixture`), the seed 1 and the Mini's
// UD-Q4_K_XL header]: thirteen layers of the Mini — the stem block whole, one whole core block of
// the loop, a coda layer of the Q8_0 expert set — of 25 visits (the core [6, 12) three passes),
// the experts narrowed to 512, the padded vocabulary of 201,024 cut to 200,058 kept.
const FIXTURE_VIEW: &[&str] = &[
    "arch Inkling2",
    "hidden 2304 vocab 201024 ctx_train 1048576",
    "rms_eps bits 0x358637bd",
    "layers 13 mtp 0",
    "hc None",
    "engram None",
    "chat pre inkling2 template bytes 6868 tools None reasoning None",
    "loop core [6, 12) passes 3 rank 64",
    "logit scale 1 softcap 30 cut 200058",
    "[0-4,6-10,12] relbias 24/8 x 128 d_rel 16 extent 1024 clips 22.627417/8 log 8192/0.1 taps 4 win 1024 gate Proj || moe 128/8 ff 512 situ 4/25 Sigmoid bias true norm true x8 hash false || attn-res-mix 6/1",
    "[5,11] relbias 24/8 x 128 d_rel 16 extent 1024 clips 22.627417/8 log 8192/0.1 taps 4 dyn gate Proj || moe 128/8 ff 512 situ 4/25 Sigmoid bias true norm true x8 hash false || attn-res-mix 6/3",
];
const FIXTURE_VISITS: &[&str] = &[
    "visits 25 over 13 stored layers",
    "runs 0-11 p0 | 6-11 p1 | 6-11 p2 | 12-12 p0",
    "windowless slots 5,11,17,23",
];
const FIXTURE_TENSORS: &[&str] = &[
    "attn bf16 17",
    "attn f32 181",
    "attn q8_0 78",
    "ffn_norm f32 39",
    "head f32 2",
    "head q8_0 1",
    "routed nvfp4 24",
    "routed q8_0 2",
    "router f32 26",
    "token_embd f32 1",
    "token_embd q8_0 1",
    "unread f32 24",
];

/// The fixture-tier clause: the planted fixture read back through the same reader — its
/// description, visits and slot table, tensors by role and type — pinned like the real files'.
/// It reads the planted path; a different file there fails on its length before the pins.
#[test]
#[ignore = "needs the inkling2 fixture on the box"]
fn hw_inkling2_fixture() {
    let mut o = String::new();
    let mut b = Vec::new();
    let path = FIXTURE.to_string();
    let len = std::fs::metadata(&path).map(|m| m.len());
    let split = Split::open(&path).unwrap_or_else(|e| {
        panic!(
            "open {path}: {e}; the fixture is planted by `fixture generate … /models/fixtures/inkling2`"
        )
    });
    let read = model::arch::spec(&split).unwrap_or_else(|e| panic!("spec of {path}: {e}"));
    let _ = writeln!(o, "fixture {path}");
    if len.as_ref().ok() != Some(&FIXTURE_BYTES) {
        b.push(format!("fixture bytes {len:?}, want {FIXTURE_BYTES}"));
    }
    spec_view::compare(
        &mut o,
        &mut b,
        "description",
        &spec_view::view(&read.spec),
        FIXTURE_VIEW,
    );
    spec_view::compare(
        &mut o,
        &mut b,
        "visits and slots",
        &visit_lines(&read.spec),
        FIXTURE_VISITS,
    );
    let mut by_role: BTreeMap<(String, String), usize> = BTreeMap::new();
    for t in &read.tensors.tensors {
        *by_role
            .entry((t.role.to_string(), t.ty.to_string()))
            .or_default() += 1;
    }
    let got: Vec<String> = by_role
        .iter()
        .map(|((role, ty), n)| format!("{role} {ty} {n}"))
        .collect();
    spec_view::compare(
        &mut o,
        &mut b,
        "tensors by role and type",
        &got,
        FIXTURE_TENSORS,
    );
    println!("{o}");
    assert!(
        b.is_empty(),
        "{} pin(s) differ:\n  {}",
        b.len(),
        b.join("\n  ")
    );
}
