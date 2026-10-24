//! solar-open2 metadata gate: what the header reader `arch::solar2` resolves from the
//! Solar-Open2-250B files, what the coverage check lists for them — the parts the program does not
//! run yet — and the plan of the Q4_K_M file on the A6000 (shape a). One contract, one test a file;
//! each prints what it compared, then fails with the whole list of what differs.
//!
//! `hw_`: needs the shards on the box (`bash tools/gate.sh --release -p bloomery-model --test
//! solar_open2_meta -- --ignored --nocapture`). Headers only: seconds.

#[path = "common/spec_fail_first.rs"]
mod spec_fail_first;
#[path = "common/spec_view.rs"]
mod spec_view;

use std::collections::BTreeMap;
use std::fmt::Write as _;

use gguf::Split;

/// The Q4_K_M file (llama.cpp's conversion, four shards): Q4_K attention and experts, Q6_K in
/// half the value projections and expert down stacks, Q5_0 low-rank KDA up projections.
const Q4KM: &str = "/models/Solar-Open2-250B-Q4_K_M/Solar-Open2-250B-Q4_K_M-00001-of-00004.gguf";

/// The IQ4_XS file (four shards): the same 1,083 tensors at other types.
const IQ4XS: &str = "/models/Solar-Open2-250B-IQ4_XS/Solar-Open2-250B-IQ4_XS-00001-of-00004.gguf";

// PIN(2026-10-20): the file's description as `arch::solar2::spec` reads it, line by line (the
// key values checked against the Q4_K_M header dump, 48 layers; `attention.head_count_kv` is 8 at
// layers 0, 4, ..., 44 and 0 elsewhere; the template is 11,721 bytes in both files). The GQA
// layers have 64 query heads, not the 32 the round brief gave: `attn_q` writes 8,192 = 64 x 128
// rows. The chat surface declares the family's parsers: its tool-call markup (`ToolFormat::TagKv`) and
// its think span.
const VIEW: &[&str] = &[
    "arch SolarOpen2",
    "hidden 4096 vocab 196608 ctx_train 1048576",
    "rms_eps bits 0x3727c5ac",
    "layers 48 mtp 0",
    "hc None",
    "engram None",
    "chat pre solar-open template bytes 11721 tools Some(TagKv) reasoning Some(ThinkSpan { open: \"<|think:start|>\", close: \"<|think:end|>\" })",
    "[0,4,8,12,16,20,24,28,32,36,40,44] gqa 64/8 x 128 rope None 0 base 0 yarn - qk_norm false out_gate Proj || moe 320/8 ff 1280 swiglu None Sigmoid bias true norm true x1 hash false shared 1280 swiglu None gate false || plain",
    "[1-3,5-7,9-11,13-15,17-19,21-23,25-27,29-31,33-35,37-39,41-43,45-47] delta Kda { gate: Softplus, beta_scale: 2.0 } k 64 v 64 d 128 conv 4 || moe 320/8 ff 1280 swiglu None Sigmoid bias true norm true x1 hash false shared 1280 swiglu None gate false || plain",
];

// PIN(2026-10-20): the file carries every key the reader reads, so it takes no default.
const DEFAULTS: &[&str] = &[];

// PIN(2026-10-20): the keys the reader does not read: listed, not refused. The dense width (no
// layer is dense) and the rope base (no layer rotates).
const UNREAD: &[&str] = &[
    "solar-open2.feed_forward_length",
    "solar-open2.rope.freq_base",
];

// PIN(2026-10-20): every tensor of the Q4_K_M file by role and type (1,083 in all) [derived] from
// the header's per-class table: attention 648 = norms and KDA parameters in f32 264 (48 + 36
// `ssm_a` + 108 conv + 36 `ssm_dt` + 36 `ssm_norm`), q4_K 288, q5_0 72 (`ssm_f_b`, `ssm_g_b`),
// q6_K 24 (half the value projections); routed experts q4_K 120 and q6_K 24 (half the down
// stacks); the shared expert the same split.
const TENSORS_Q4KM: &[&str] = &[
    "attn f32 264",
    "attn q4_K 288",
    "attn q5_0 72",
    "attn q6_K 24",
    "ffn_norm f32 48",
    "head f32 1",
    "head q6_K 1",
    "routed q4_K 120",
    "routed q6_K 24",
    "router f32 96",
    "shexp q4_K 120",
    "shexp q6_K 24",
    "token_embd q4_K 1",
];

// PIN(2026-10-20): the same for the IQ4_XS file [derived]: attention q6_K 204 (the 12 gates and
// four of the five 48-layer projections), `ssm_f_b`/`ssm_g_b` iq4_nl 72, the other low-rank
// matrices iq4_xs 108; routed experts iq4_xs 141 and q5_K 3 (one layer); the shared expert iq4_xs.
const TENSORS_IQ4XS: &[&str] = &[
    "attn f32 264",
    "attn iq4_nl 72",
    "attn iq4_xs 108",
    "attn q6_K 204",
    "ffn_norm f32 48",
    "head f32 1",
    "head q6_K 1",
    "routed iq4_xs 141",
    "routed q5_K 3",
    "router f32 96",
    "shexp iq4_xs 144",
    "token_embd q6_K 1",
];

/// One file against the pins: the description, the defaults, the unread keys, the tensors by role
/// and type, the coverage check's list and its FAIL-first `specimens`.
fn compare_file(path: &str, tensors: &[&str], specimens: &[(&str, &str)]) {
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
        VIEW,
    );
    spec_view::compare(&mut o, &mut b, "defaults taken", &read.defaults, DEFAULTS);
    let unread = model::arch::solar2::hparams::unread_keys(&split);
    spec_view::compare(&mut o, &mut b, "keys not read", &unread, UNREAD);
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
    let list = model::arch::coverage::check(&read.spec, &read.tensors);
    spec_view::compare(
        &mut o,
        &mut b,
        "coverage",
        &spec_view::items(&list),
        COVERAGE,
    );
    for (at, want) in specimens {
        spec_fail_first::fail_first(&mut o, &mut b, &read.spec, &read.tensors, at, want);
    }
    println!("{o}");
    assert!(
        b.is_empty(),
        "{} pin(s) differ:\n  {}",
        b.len(),
        b.join("\n  ")
    );
}

// PIN(2026-10-20): the three items that stood after the program (the 320-expert router instance,
// the `solar-open` pre-tokenizer, the tool parser) left the list: so3's `sigmoid_320_8` covers the
// router, the tokenizer's `solar-open` row (Qwen2's split) the pre-tokenizer, and `spec_of`'s binding
// of the family's markup and think span (`chat.tools`, `chat.reasoning`) the parser. Re-pin by
// specification, not a relaxation. FAIL-first: the binding removed lists the parser item again.
// PIN(2026-10-20): the coverage check's list once `Program::Solar2Body` runs the family (it was
// ten items with no program: the program itself, the gate projection, the softplus-decay KDA, the
// q5_0 low-rank matrices and, for IQ4_XS, every type no card format loads). The program's rows
// cover the GQA flash (head 128, group 8), the gate projection, the softplus-decay β-2 KDA, the
// shared expert, the recurrent state and the mixed trunk; its routed arm is the common leg's
// (`kernels::card_runs`, else a host qdot kernel: both files pass, the Q6_K downs and the IQ4_XS
// stacks at k 1,280 and 4,096 among them), its token embedding the host's rows. What is left is
// the router (the shape table has no 320-expert instance until so3 adds `sigmoid_320_8`), the
// `solar-open` pre-tokenizer the tokenizer does not know and the tool parser of a template with
// no declared markup. The program pins no other tensor's type: the body's typed sites refuse a
// type they do not launch at load, by name (`site::file_site`).
const COVERAGE: &[&str] = &[];

// PIN(2026-10-20): the check's FAIL-first specimen: with the row at `.0` removed from the table the
// check lists `.1`, the need that row covers, on its layers. Mutant: `Program::Solar2Body` left
// out of that row.
const SPECIMENS: &[(&str, &str)] = &[
    (
        "gpu/src/linear/mixer_kda.rs (kda_conv_prep, kda_delta with the softplus decay and the β scale)",
        "delta rule KDA: d 128, 64 heads, conv 4, softplus gate, beta scaled by 2: 1-3,5-7,9-11,13-15,17-19,21-23,25-27,29-31,33-35,37-39,41-43,45-47",
    ),
    (
        "gpu/src/arch/hybrid.rs GqaGate::Proj (the out gate of the layer's own projection)",
        "attention output gate: sigmoid of its own projection: 0,4,8,12,16,20,24,28,32,36,40,44",
    ),
    (
        "gpu-glm5next/src/ffn.rs (the shared expert)",
        "shared expert, ff 1280: 0-47",
    ),
];

#[test]
#[ignore = "needs the Solar-Open2-250B Q4_K_M shards on the box"]
fn hw_solar_open2_spec_q4_k_m() {
    compare_file(Q4KM, TENSORS_Q4KM, SPECIMENS);
}

#[test]
#[ignore = "needs the Solar-Open2-250B IQ4_XS shards on the box"]
fn hw_solar_open2_spec_iq4_xs() {
    compare_file(IQ4XS, TENSORS_IQ4XS, &[]);
}

// PIN(2026-10-20): the plan of the Q4_K_M file with the A6000 running every layer and the experts
// on the host (plan shape a, `workstation::plan_a`), 32,768 positions, one slot, the room this
// workstation has with nothing loaded. The stores are derived from the description alone [derived]:
// a GQA layer 8 key heads x (128 + 128) x 2 B = 4,096 B a position, 12 layers; a KDA layer 64 x
// 128 x 128 f32 (4,194,304 B) + the conv ring 11 rows x 3 x 64 x 128 f32 (1,081,344 B) + one
// lane's u32 stamp = 5,275,652 B, 36 layers; no rope rows (NoPE). Mutants: the stamp or the pass
// rows of the ring left out, a window ring on a full layer, a rope table counted.
// The experts: the common card leg has no Q6_K down arm, so the 24 layers whose down stacks are
// Q6_K (TENSORS_Q4KM) keep all their experts on the host and the other 24 are the card's candidates.
#[test]
#[ignore = "needs the Solar-Open2-250B Q4_K_M shards on the box"]
fn hw_solar_open2_plan_a() {
    use model::arch::solar2::place::{PlaceError, PlanInputs};
    use model::placement::kernels::card_runs;
    use model::placement::workstation::{self, TierBatchBytes};
    use model::placement::{PlanLevers, Role};

    const CTX: u64 = 32_768;
    const GQA_LAYERS: u64 = 12;
    const KDA_LAYERS: u64 = 36;
    const GQA_POSITION: u64 = 8 * (128 + 128) * 2;
    const KDA_STATE: u64 = 4_194_304 + 1_081_344 + 4;
    const CANDIDATES: usize = 24;

    let mut o = String::new();
    let mut b = Vec::new();
    let split = Split::open(Q4KM).unwrap_or_else(|e| panic!("open {Q4KM}: {e}"));
    let mut inputs =
        PlanInputs::describe(&split).unwrap_or_else(|e| panic!("plan inputs of {Q4KM}: {e}"));
    // The room is this workstation's with nothing loaded, given: the pins below hold the unpaged
    // plan, whatever the box's `MemAvailable` reads now.
    inputs.room = (workstation::PLAN_ROOM, workstation::HostRead::Given);
    let layers = inputs.spec.layers.len();
    let _ = writeln!(o, "file {Q4KM}, {layers} layers, ctx {CTX}");
    row(
        &mut o,
        &mut b,
        "kv bytes a position over the trunk",
        inputs.kv.bytes(0..layers, 1),
        GQA_LAYERS * GQA_POSITION + KDA_LAYERS * KDA_STATE,
    );
    let kv_want = GQA_LAYERS * GQA_POSITION * CTX + KDA_LAYERS * KDA_STATE;
    row(
        &mut o,
        &mut b,
        "kv bytes at the context",
        inputs.kv.bytes(0..layers, CTX),
        kv_want,
    );
    // PIN(2026-10-20): the prompt batch's reserve, 558,471,268 B [derived] from the file's widths (4,096
    // wide; 64 query and 8 key heads of 128; 64 KDA heads of 128, so qkv rows of 24,576; 320 experts,
    // 8 used, 1,280 wide, one shared of 1,280) at 512 rows, four bytes a value, term by term:
    // step rows 273,416,192 (a row 4 x 4,096 + 2 x 24,576 + 2 x 128 + 2 x 64 + 5 x 8,192 + 3 x 8,192 +
    // 2 x 1,024 values) + host sums and qkv staging 58,720,256 (4,096 + 2 x 8,192 + 8,192 a row) +
    // routed rows 42,647,552 (3 x 4,096 + 320 + 2 x 8 + 2 x 4,096 + 8 a row) + the shared expert's gate
    // and up 5,242,880 (2 x 1,280 a row) + the flash's partials 21,299,200 (8 rows x 64 heads x 80
    // segments x 130 values) + the position, key-count and word records 4,132 + one column of each
    // activation form 55,200 + the wide forms 40,587,264 (the widths 4,096 twice, 8,192, 128 and 1,280,
    // both forms where the width allows them) + two routes 8,352 + the card leg 116,490,240
    // (`tier_chunk_scratch_bytes` at 4,096 x 1,280, 8 used, 512 columns). Mutant: a term left out, or
    // the rows counted at 256.
    const RESERVE: u64 = 558_471_268;
    row(
        &mut o,
        &mut b,
        "prompt reserve bytes",
        inputs.prompt_reserve,
        RESERVE,
    );
    let machine = workstation::plan_a(layers);
    let plan = inputs
        .plan(&machine, CTX, &PlanLevers::default())
        .unwrap_or_else(|e| panic!("plan on the A6000 at ctx {CTX}: {e}"));
    let card = &plan.cards[0];
    row(&mut o, &mut b, "card kv bytes", card.kv_bytes, kv_want);
    row(
        &mut o,
        &mut b,
        "card stage rows bytes",
        card.stage_rows_bytes,
        0,
    );
    // A layer is a candidate when each of its routed stacks is of a type the card leg runs.
    let candidate = |l: usize| {
        inputs
            .model
            .tensors
            .iter()
            .filter(|t| t.role == Role::RoutedExperts && t.layer == Some(l))
            .all(|t| card_runs(t.ty))
    };
    let candidates: Vec<usize> = (0..layers).filter(|&l| candidate(l)).collect();
    row(
        &mut o,
        &mut b,
        "layers the card leg can run",
        candidates.len(),
        CANDIDATES,
    );
    let held = |l: usize| plan.n_l[l];
    row(
        &mut o,
        &mut b,
        "no expert on the card in a Q6_K-down layer",
        (0..layers).filter(|&l| !candidate(l)).all(|l| held(l) == 0),
        true,
    );
    let on_card: Vec<u64> = candidates.iter().map(|&l| held(l)).collect();
    let (most, least) = (on_card.iter().max(), on_card.iter().min());
    row(
        &mut o,
        &mut b,
        "candidate layers hold even counts",
        most.zip(least).is_some_and(|(m, l)| *l >= 1 && m - l <= 1),
        true,
    );
    let n: u64 = plan.n_l.iter().sum();
    // PIN(2026-10-20): the reserve displaces card experts: the same plan with no reserve holds
    // `RESERVE / 8,847,360` more [derived]: an expert is a Q4_K gate, up and down of 4,096 x 1,280 values,
    // 3 x 4,096 x 1,280 x 144 / 256 = 8,847,360 B, so 558,471,268 / 8,847,360 = 63.12 and the count
    // drops by 63 or 64 (the budget's remainder decides which). Mutant: the reserve left out of the
    // plan (the drop is 0).
    let unreserved = inputs.clone().with_prompt_reserve(0);
    let free = unreserved
        .plan(&machine, CTX, &PlanLevers::default())
        .unwrap_or_else(|e| panic!("plan with no reserve at ctx {CTX}: {e}"));
    let n_free: u64 = free.n_l.iter().sum();
    let _ = writeln!(
        o,
        "  card experts with no reserve {n_free}, with the reserve {n}: {} fewer",
        n_free.saturating_sub(n)
    );
    row(
        &mut o,
        &mut b,
        "card experts the reserve displaces",
        (63..=64).contains(&n_free.saturating_sub(n)),
        true,
    );
    let _ = writeln!(
        o,
        "  n_l on candidate layers {on_card:?}\n  card experts {n} of {} ({:.1} %), card dense bytes {}, \
         card expert bytes {}, host routed bytes {}",
        inputs.model.experts * layers as u64,
        100.0 * n as f64 / (inputs.model.experts * layers as u64) as f64,
        card.dense_bytes,
        card.expert_bytes,
        plan.host.expert_bytes
    );
    // PIN(2026-10-21): plan shape bp hangs the 3090 as an expert tier, which the body serves
    // ([`PlanInputs::tier_reserve`], the places of the prompt batch): the stage card keeps
    // 3 x 4 x 512 x 8 = 49,152 B more than plan a [derived: a place, the join's copy of it and of the
    // slot's weight, four bytes each, for each of 8 used experts of 512 rows], which displaces at
    // most one expert of 8,847,360 B [derived above], so the stage holds `n` or `n - 1` experts;
    // the tier holds experts on candidate layers only, of the same size, at most a layer's 320
    // with the stage's, and its bytes are its experts' times that size. Mutant: the body's tier
    // reserve left out of `describe` (the plan is refused by name), or the tier planned on a
    // layer the card leg cannot run.
    let tier_batch = TierBatchBytes {
        staging: 0,
        scratch: 0,
        host: 0,
        leg: None,
    };
    let bp = workstation::plan_bp(layers, None, tier_batch);
    let tiered = inputs
        .plan(&bp, CTX, &PlanLevers::default())
        .unwrap_or_else(|e| panic!("plan bp at ctx {CTX}: {e}"));
    let tier_n = &tiered.tier_n_l[0];
    let tier_experts: u64 = tier_n.iter().sum();
    let stage_experts: u64 = tiered.n_l.iter().sum();
    let _ = writeln!(
        o,
        "  plan bp: stage experts {stage_experts}, tier experts {tier_experts} ({} B), host experts {}; \
         the stage reserve {} B",
        tiered.cards[1].expert_bytes,
        tiered.host.experts,
        inputs.reserve_on(&bp).unwrap_or(0)
    );
    row(
        &mut o,
        &mut b,
        "plan bp: the stage's reserve is plan a's and the tier places",
        inputs.reserve_on(&bp).ok(),
        Some(RESERVE + 3 * 4 * 512 * 8),
    );
    row(
        &mut o,
        &mut b,
        "plan bp: the tier holds experts",
        tier_experts > 0,
        true,
    );
    row(
        &mut o,
        &mut b,
        "plan bp: the stage holds plan a's experts less at most one",
        (n.saturating_sub(1)..=n).contains(&stage_experts),
        true,
    );
    row(
        &mut o,
        &mut b,
        "plan bp: the tier only on layers the card leg can run, a layer's experts at most",
        (0..layers).all(|l| {
            let t = tier_n[l];
            t == 0 || (candidate(l) && tiered.n_l[l] + t <= inputs.model.experts)
        }),
        true,
    );
    row(
        &mut o,
        &mut b,
        "plan bp: the tier's expert bytes",
        tiered.cards[1].expert_bytes,
        tier_experts * 8_847_360,
    );
    // The body that hangs no tier card refuses the same machine by name, the card and the missing
    // tier named.
    let mut untiered = inputs.clone();
    untiered.tier_reserve = None;
    let err = untiered
        .plan(&bp, CTX, &PlanLevers::default())
        .expect_err("a body that hangs no expert tier refuses a machine that names one");
    let _ = writeln!(o, "  as a body with no tier refuses shape bp: {err}");
    if !matches!(err, PlaceError::Tier { tier: 0, .. })
        || !(err.to_string().contains("expert tier card 3090")
            && err.to_string().contains("hangs no expert tier"))
    {
        b.push(format!(
            "the tier refusal is not PlaceError::Tier on the 3090: {err}"
        ));
    }
    let err = inputs
        .plan(&machine, 0, &PlanLevers::default())
        .expect_err("a context of no position is refused by name");
    if !matches!(err, PlaceError::CtxZero) {
        b.push(format!("the ctx-0 refusal: {err}"));
    }
    println!("{o}");
    assert!(
        b.is_empty(),
        "{} pin(s) differ:\n  {}",
        b.len(),
        b.join("\n  ")
    );
}

/// One compared value: a line of the printed table, and an entry of `bad` when it differs.
fn row<T: PartialEq + std::fmt::Debug>(
    out: &mut String,
    bad: &mut Vec<String>,
    what: &str,
    got: T,
    want: T,
) {
    let mark = if got == want { "ok " } else { "BAD" };
    let _ = writeln!(out, "  {mark} {what:<34} {got:?}");
    if got != want {
        bad.push(format!("{what}: got {got:?}, want {want:?}"));
    }
}
