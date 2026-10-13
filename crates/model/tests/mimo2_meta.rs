//! mimo2 metadata gate: what the header reader `arch::mimo2` resolves from
//! the MiMo-V2.6-Flash-RL file, and what the coverage check lists for it —
//! the parts no program in this tree runs; empty while `gpu-mimo2` runs the
//! file whole. One contract, one test; it prints what it compared, then
//! fails with the whole list of what differs.
//!
//! `hw_`: needs the RL shards on the box (`just gate-mimo2-meta`). Headers
//! only: seconds.

#[path = "common/spec_view.rs"]
mod spec_view;

use std::collections::BTreeMap;
use std::fmt::Write as _;

use gguf::{Gguf, Split, Value};

/// The file: ggml's RL conversion, MXFP4 experts and a Q8_0 dense tier. Its
/// first shard holds the header and no tensor.
const MIMO: &str = "/models/MiMo-V2.6-Flash-RL/MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf";

/// The separate MTP file, whose 51 blocks carry only the next-token three:
/// the target reader refuses it by a trunk tensor it does not carry; the
/// sidecar reader (`mimo2::mtp::mtp_of`) reads it beside a target.
const MTP: &str = "/models/MiMo-V2.6-Flash-RL/mtp-MiMo-V2.6-Flash-RL-MXFP4.gguf";

/// The MOPD checkpoint's target (the file the ik oracle is dumped from), its
/// own Q8_0 sidecar, and the RL checkpoint's Q8_0 sidecar as the type
/// census's header stub (`tools/type-census.py build`, which writes
/// `<root>/<org>/<repo>/<file>`: a sparse file of the real header, read by
/// header alone).
const MOPD: &str = "/models/MiMo-V2.6-Flash-MOPD/MiMo-V2.6-Flash-MOPD-MXFP4-00001-of-00002.gguf";
const MOPD_MTP: &str = "/models/MiMo-V2.6-Flash-MOPD/mtp-MiMo-V2.6-Flash-MOPD-Q8_0.gguf";
const RL_MTP_STUB: &str = "/models/census-stubs-a134239f/ggml-org/MiMo-V2.6-Flash-RL-GGUF/mtp-MiMo-V2.6-Flash-RL-Q8_0.gguf";

// PIN(2026-10-07): the file's description as `arch::mimo2::spec` reads it, line by line (the key
// values checked against the shard-0 header dump and llama.cpp's mimo2 loader, src/models/mimo2.
// cpp). The value head is narrower than the key head (192/128), the KV-head count and the rope
// base are the layer's own (4 and 1e7 on the nine full layers, 8 and 1e4 on the window ones).
const VIEW: &[&str] = &[
    "arch MiMo2",
    "hidden 4096 vocab 152576 ctx_train 1048576",
    "rms_eps bits 0x358637bd",
    "layers 48 mtp 0",
    "hc None",
    "engram None",
    "chat pre qwen2 template bytes 3867 tools Some(QwenXml) reasoning Some(ThinkSpan)",
    "[0] gqa 64/4 x 192 v 128 rope Neox 64 base 10000000 yarn - qk_norm false out_gate false vscale 0.707 || dense 16384 swiglu None || plain",
    "[5,11,17,23,29,35,41,47] gqa 64/4 x 192 v 128 rope Neox 64 base 10000000 yarn - qk_norm false out_gate false vscale 0.707 || moe 256/8 ff 2048 swiglu None Sigmoid bias true norm true x1 hash false || plain",
    "[1-4,6-10,12-16,18-22,24-28,30-34,36-40,42-46] gqa 64/8 x 192 v 128 rope Neox 64 base 10000 yarn - qk_norm false out_gate false win 128 sinks vscale 0.707 || moe 256/8 ff 2048 swiglu None Sigmoid bias true norm true x1 hash false || plain",
];

// PIN(2026-10-07): the RL file carries no next-token layer; the separate `mtp-` file does.
const MTP_LAYERS: &[&str] = &[];

// PIN(2026-10-07): two keys the loader reads with a fallback (absent nextn count; the scale node
// the graph skips at 1).
const DEFAULTS: &[&str] = &[
    "nextn_predict_layers = 0 (absent)",
    "expert_weights_scale = 1 (absent, no scale node)",
];

// PIN(2026-10-07): the keys the reader does not read: listed, not refused.
const UNREAD: &[&str] = &["general.sampling.temp", "general.sampling.top_p"];

// PIN(2026-10-07): every tensor by role and type (472 in all, none in the first shard).
const TENSORS: &[&str] = &[
    "attn f32 87",
    "attn q8_0 96",
    "dense_ffn q8_0 3",
    "ffn_norm f32 48",
    "head f32 1",
    "head q8_0 1",
    "routed mxfp4 141",
    "router f32 94",
    "token_embd q8_0 1",
];

// PIN(2026-10-07): the KV cache the description asks a layer to hold, in values a position: the
// full layers' 4 heads of 320 (key 192 + value 128; 2 B/value in the f16 cache) and the window
// layers' 8. The values a position stay these. How many rows a window layer holds is the ring
// pin on the plan, not this list.
const KV: &[&str] = &[
    "kv 4x320 values: 0,5,11,17,23,29,35,41,47",
    "kv 8x320 values: 1-4,6-10,12-16,18-22,24-28,30-34,36-40,42-46",
];

// PIN(2026-10-08): the coverage check's list (feature: layers), empty. `crates/gpu-mimo2` runs the
// file whole, so each need of its 48 layers has a row naming the program: the flashes at head 192
// over value 128 (group 16 on the nine full layers, group 8 with window 128 and sinks on the 39
// window layers), the rope without a QK norm at 64 of 192 dims, the BiasedSigmoid router at 256
// experts and the dense layer 0 are rows; every tensor type is one the body reads (Q8_0 matrices,
// F32 vectors) and the 141 mxfp4 stacks are served by the host's fused kernel at their row widths
// (4096 and 2048). Before the program the list held seven items: the two flashes, the rope, three
// mxfp4 rows and the program itself.
const COVERAGE: &[&str] = &[];

#[test]
#[ignore = "needs the MiMo-V2.6-Flash-RL shards on the box (just gate-mimo2-meta)"]
fn hw_mimo2_spec() {
    let mut o = String::new();
    let mut b = Vec::new();
    let split = Split::open(MIMO).unwrap_or_else(|e| panic!("open {MIMO}: {e}"));
    let read = model::arch::spec(&split).unwrap_or_else(|e| panic!("spec of {MIMO}: {e}"));
    let _ = writeln!(o, "file {MIMO}");
    spec_view::compare(
        &mut o,
        &mut b,
        "description",
        &spec_view::view(&read.spec),
        VIEW,
    );
    let n_trunk = read.spec.layers.len();
    let mtp: Vec<String> = (n_trunk..)
        .zip(&read.spec.mtp)
        .map(|(l, layer)| format!("mtp {l}: {}", spec_view::layer_line(layer)))
        .collect();
    spec_view::compare(&mut o, &mut b, "next-token layers", &mtp, MTP_LAYERS);
    spec_view::compare(&mut o, &mut b, "defaults taken", &read.defaults, DEFAULTS);
    let unread = model::arch::mimo2::hparams::unread_keys(&split);
    spec_view::compare(&mut o, &mut b, "keys not read", &unread, UNREAD);
    let mut by_role: BTreeMap<(String, String), usize> = BTreeMap::new();
    for t in &read.tensors.tensors {
        *by_role
            .entry((t.role.to_string(), t.ty.to_string()))
            .or_default() += 1;
    }
    let tensors: Vec<String> = by_role
        .iter()
        .map(|((role, ty), n)| format!("{role} {ty} {n}"))
        .collect();
    let _ = writeln!(o, "tensors {}", read.tensors.tensors.len());
    spec_view::compare(
        &mut o,
        &mut b,
        "tensors by role and type",
        &tensors,
        TENSORS,
    );
    let kv = kv_values(&read.spec);
    spec_view::compare(&mut o, &mut b, "kv values a layer holds", &kv, KV);
    let list = model::arch::coverage::check(&read.spec, &read.tensors);
    spec_view::compare(
        &mut o,
        &mut b,
        "coverage",
        &spec_view::items(&list),
        COVERAGE,
    );
    // PIN(2026-10-13): a prompt call writes every position on the nine full layers and only the
    // window on the other thirty-nine. The full layers are 0, 5, 11, 17, 23, 29, 35, 41, 47.
    // Mapping a window layer's GQA store back to a positional store turns this list into every layer.
    let full = vec![0usize, 5, 11, 17, 23, 29, 35, 41, 47];
    let every: Vec<usize> = read
        .spec
        .layers
        .iter()
        .enumerate()
        .filter(|(_, layer)| runtime::state::every_position(layer))
        .map(|(i, _)| i)
        .collect();
    row(
        &mut o,
        &mut b,
        "layers written at every position",
        every,
        full,
    );
    println!("{o}");
    assert!(
        b.is_empty(),
        "{} pin(s) differ:\n  {}",
        b.len(),
        b.join("\n  ")
    );
}

// PIN(2026-10-08): the plan of the RL file through `arch::mimo2::place::PlanInputs` on the gate
// machine (the 3090, all 48 trunk layers and the head) at the gate context 4160, made from
// `PlanInputs::describe`. KV a position over the trunk 222,720 B = (39 window × 8 + 9 full × 4) KV heads ×
// 320 values (key 192 + value 128) × 2 B, every layer a full plane.
// PIN(2026-10-08): the card's dense weights 5,839,528,704 B [derived] = attention Q8_0 9 × 94,699,520 +
// 39 × 100,270,080 (a layer's rows: 4,096 × (64·192 + kv·(192 + 128)) of qkv + 64·128 × 4,096 of
// o, × 34/32) + the dense block 3 × 4,096 × 16,384 × 34/32 + the head 152,576 × 4,096 × 34/32 +
// the routers 47 × 256 × 4,096 × 4 + the norms, biases and sinks 97 × 4,096 × 4 + 47 × 256 × 4 +
// 39 × 64 × 4. The embedding 664,010,752 B = 152,576 × 4,096 × 34/32 on the host as the
// row-gathered table.
// PIN(2026-10-10): an expert is 13,369,344 B = 3 stacks × 4,096 × 2,048 values × 17/32 B (MXFP4), the
// routed experts 47 × 256 = 12,032 and 160,859,947,008 B [derived]. The card experts run
// (`place::card_routed`: the MXFP4 stacks the common pair reads), so the plan puts on the card as many
// as its budget holds, one at a time on the routed layer with the fewest (the first on a tie): the dense
// layer 0 none, the routed layers' counts non-increasing and within one of each other. The count N
// is bounded by the budget arithmetic [derived]: the card's usable bytes less 1,600 MiB (context 512,
// scratch 64, margin 1,024), the KV planes (222,720 B × 4,160) and the dense weights, over an expert's
// bytes — at most ⌊16,906,607,872 / 13,369,344⌋ = 1,264 on the 3090's bytes (usable 24,176 MiB) and
// ⌊42,508,639,488 / 13,369,344⌋ = 3,179 on the A6000 (usable 48,592 MiB), and at the least 61
// experts fewer (the allocator's 2 MiB granules over the 391 or so uploads, 782 MiB at their worst):
// 1,203 and 3,118. The count the plan makes is printed, and the card's and the host's expert bytes
// are N and 12,032 − N experts of the bytes above.
// PIN(2026-10-13): the engine default is the ring. A full layer is 23,040 B a position
// (9 × 4 heads × 320 × 2) and a window layer 199,680 B (39 × 8 × 320 × 2). At one position
// the ring is the context, so the 222,720 B pin holds. At context 4,160 the ring is 768 rows
// (`ring_rows(128, 4160)`), and one sequence is 4,160 × 23,040 + 768 × 199,680 = 249,200,640 B
// [derived]. The plane term stays 4,160 × 222,720 = 926,515,200 B. The window bytes the ring
// frees are (4,160 − 768) × 199,680 = 677,314,560, which is 50 experts of 13,369,344 plus a
// remainder 8,847,360. That remainder crosses one more expert on both of the 2026-10-10
// numerators, so the floors rise by 51, to 1,315 and 3,230, not by 50. The granule band stays
// the same 61: the ring shrinks the window buffers and does not drop an upload, so the lows
// are 1,254 and 3,169. The plane bounds above are not replaced.
/// The plan of the RL file onto the gate machine and onto the A6000's: what
/// `arch::mimo2::place::PlanInputs` resolves for the file the spec test
/// reads — the routed experts on the card as far as its budget holds them
/// (every routed layer's prefix) and the rest on the host. The engine default
/// keeps a window layer on the ring and a full layer on the plane; the same
/// plan with `Rows::Plane` keeps today's full-plane KV pins. The named
/// refusals stay: a machine that hangs an expert tier card, and a context of
/// no position.
#[test]
#[ignore = "needs the MiMo-V2.6-Flash-RL shards on the box (just gate-mimo2-meta)"]
fn hw_mimo2_plan() {
    use model::placement::PlanLevers;
    use model::placement::workstation;

    /// An expert's bytes, and the routed experts of the file.
    const EXPERT_BYTES: u64 = 13_369_344;
    const ROUTED: u64 = 47 * 256;

    let mut o = String::new();
    let mut b = Vec::new();
    let split = Split::open(MIMO).unwrap_or_else(|e| panic!("open {MIMO}: {e}"));
    let mut inputs = model::arch::mimo2::place::PlanInputs::describe(&split)
        .unwrap_or_else(|e| panic!("plan inputs of {MIMO}: {e}"));
    // The room is this workstation's with nothing loaded, given: the pins below hold the
    // unpaged plan, whatever the box's `MemAvailable` reads now.
    inputs.room = (workstation::PLAN_ROOM, workstation::HostRead::Given);
    let layers = inputs.hp.n_trunk;
    let ctx: u64 = 4160;
    let _ = writeln!(o, "file {MIMO}, {layers} trunk layers, ctx {ctx}");
    // `describe` leaves the ring on. At one position that ring is the context, so dropping
    // `ring_rows`'s `min(ctx, …)` turns this pin red. At the gate context the ring is the
    // two-term figure, and the plane term is today's product.
    row(
        &mut o,
        &mut b,
        "kv bytes a position over the trunk",
        inputs.kv.bytes(0..layers, 1),
        222_720,
    );
    row(
        &mut o,
        &mut b,
        "ring kv bytes at the gate context",
        inputs.kv.bytes(0..layers, ctx),
        249_200_640,
    );
    let mut plane = inputs.clone();
    plane.kv = model::arch::mimo2::place::KvLayout::with_rows(
        &inputs.hp,
        model::arch::mimo2::place::Rows::Plane,
    );
    row(
        &mut o,
        &mut b,
        "plane kv bytes at the gate context",
        plane.kv.bytes(0..layers, ctx),
        ctx * 222_720,
    );
    {
        let mut check = |term: &model::arch::mimo2::place::PlanInputs,
                         which: &str,
                         kv_want: u64,
                         gate_bound: std::ops::RangeInclusive<u64>,
                         a6000_bound: std::ops::RangeInclusive<u64>| {
            for (name, machine, bound) in [
                (
                    format!("gate machine (the 3090's bytes), {which}"),
                    workstation::plan_gate(layers),
                    gate_bound,
                ),
                (
                    format!("A6000 (the default place), {which}"),
                    workstation::plan_a(layers),
                    a6000_bound,
                ),
            ] {
                let plan = term
                    .plan(&machine, ctx, &PlanLevers::default())
                    .unwrap_or_else(|e| panic!("plan on the {name} at ctx {ctx}: {e}"));
                let card = &plan.cards[0];
                let n: u64 = plan.n_l.iter().sum();
                let _ = writeln!(o, "{name}: n_l {:?}", plan.n_l);
                row(
                    &mut o,
                    &mut b,
                    "card dense bytes",
                    card.dense_bytes,
                    5_839_528_704,
                );
                row(&mut o, &mut b, "card kv bytes", card.kv_bytes, kv_want);
                let held = (bound.contains(&n), format!("{n} in {bound:?}"));
                row(
                    &mut o,
                    &mut b,
                    "card experts inside the budget's bound",
                    held.0,
                    true,
                );
                let _ = writeln!(o, "  card experts {}", held.1);
                let spread = plan.n_l[0] == 0
                    && plan.n_l[1..].windows(2).all(|w| w[0] >= w[1])
                    && plan.n_l[1] - plan.n_l[layers - 1] <= 1
                    && plan.n_l[layers - 1] >= 1;
                row(
                    &mut o,
                    &mut b,
                    "dense layer none, routed counts even",
                    spread,
                    true,
                );
                row(
                    &mut o,
                    &mut b,
                    "card expert bytes",
                    card.expert_bytes,
                    n * EXPERT_BYTES,
                );
                row(
                    &mut o,
                    &mut b,
                    "host routed bytes",
                    plan.host.expert_bytes,
                    (ROUTED - n) * EXPERT_BYTES,
                );
                row(
                    &mut o,
                    &mut b,
                    "host table bytes (the embedding)",
                    plan.host.table_bytes,
                    664_010_752,
                );
            }
        };
        check(&plane, "plane", ctx * 222_720, 1_203..=1_264, 3_118..=3_179);
        check(&inputs, "ring", 249_200_640, 1_254..=1_315, 3_169..=3_230);
    }
    let machine = workstation::plan_gate(layers);
    let mut tiered = workstation::plan_gate(layers);
    let mut tier = tiered.cards[0].clone();
    tier.layers = 0..0;
    tier.head = false;
    tier.token_embedding = false;
    tiered.tiers.push(tier);
    let err = inputs
        .plan(&tiered, ctx, &PlanLevers::default())
        .expect_err("a machine that hangs an expert tier card is refused by name")
        .to_string();
    let _ = writeln!(o, "  as the plan refuses a tier card: {err}");
    // PIN(2026-10-10): the tier refusal is the program's own (`PlaceError::Tier`): it names the card
    // and that the program hangs no expert tier on its card experts.
    if !(err.contains("expert tier card 3090") && err.contains("hangs no expert tier")) {
        b.push(format!(
            "the tier refusal names neither the card nor the missing tier: {err}"
        ));
    }
    let err = inputs
        .plan(&machine, 0, &PlanLevers::default())
        .expect_err("a context of no position is refused by name")
        .to_string();
    let _ = writeln!(o, "  as the plan refuses ctx 0: {err}");
    if !err.contains("ctx_max 0") {
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

/// One compared value: a line of the printed table, and an entry of `bad`
/// when it differs.
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

/// Each run of layers that hold one KV width, in values a position: the
/// KV-head count times the key and value head widths.
fn kv_values(spec: &model::arch::models::ModelSpec) -> Vec<String> {
    use model::arch::models::Mixer;
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    for (l, layer) in spec.layers.iter().enumerate() {
        let Mixer::Gqa(g) = &layer.mixer else {
            panic!("a mimo2 layer that is not GQA");
        };
        let line = format!("kv {}x{} values", g.kv_heads, g.head_dim + g.value_dim);
        match groups.iter_mut().find(|(k, _)| *k == line) {
            Some((_, ls)) => ls.push(l),
            None => groups.push((line, vec![l])),
        }
    }
    groups
        .into_iter()
        .map(|(line, ls)| format!("{line}: {}", spec_view::ranges_of(&ls)))
        .collect()
}

/// A doctored header — the RL shard-0's own keys with the KV-head array one
/// short — is refused by the array's name: the per-layer arrays are the
/// file's fact, and a wrong length is never read as a broadcast.
#[test]
#[ignore = "needs the MiMo-V2.6-Flash-RL shards on the box (just gate-mimo2-meta)"]
fn hw_mimo2_short_kv_array_refused() {
    let shard0 = Gguf::open(MIMO).expect("the metadata shard opens");
    let kv = doctored_kv(&shard0, "mimo2.attention.head_count_kv", 47);
    let path = std::env::temp_dir().join(format!(
        "bloomery-mimo2-short-kv-{}.gguf",
        std::process::id()
    ));
    std::fs::write(&path, kv).expect("write the doctored header");
    let split = Split::open(&path).expect("the doctored header opens");
    let err = model::arch::spec(&split)
        .expect_err("a short per-layer array is refused")
        .to_string();
    let _ = std::fs::remove_file(&path);
    assert!(
        err.contains("mimo2.attention.head_count_kv")
            && err.contains("has 47 values for 48 layers"),
        "{err}"
    );
}

/// The separate MTP file, whose blocks carry only the next-token layers, is
/// refused by the target reader at the first trunk tensor it lacks; the
/// sidecar reader opens it beside a target
/// (`hw_mimo2_mtp_sidecar_reads_beside_its_checkpoints_target`).
#[test]
#[ignore = "needs the MiMo-V2.6-Flash-RL mtp file on the box (just gate-mimo2-meta)"]
fn hw_mimo2_mtp_file_is_refused_until_its_round() {
    let split = Split::open(MTP).unwrap_or_else(|e| panic!("open {MTP}: {e}"));
    let err = model::arch::spec(&split)
        .expect_err("a next-token-only file has no trunk")
        .to_string();
    assert!(
        err.contains("tensor blk.0.attn_norm.weight: is not in the file"),
        "{err}"
    );
}

/// The MOPD target's sidecar of its own checkpoint reads as one next-token
/// layer, `blk.48`: a window layer with sinks over the target's dense block,
/// fed the rows before the norms, the two matrices carried.
#[test]
#[ignore = "needs the MiMo-V2.6-Flash-MOPD shards and its Q8_0 sidecar on the box (just gate-mimo2-meta)"]
fn hw_mimo2_mtp_sidecar_reads_beside_its_checkpoints_target() {
    use model::arch::models::{DraftSpec, Ffn, Mixer, MtpHeadNorm, MtpInput, MtpSource};

    let target = Split::open(MOPD).unwrap_or_else(|e| panic!("open {MOPD}: {e}"));
    let read = model::arch::spec(&target).unwrap_or_else(|e| panic!("spec of {MOPD}: {e}"));
    let draft = Split::open(MOPD_MTP).unwrap_or_else(|e| panic!("open {MOPD_MTP}: {e}"));
    let spec = model::arch::mimo2::mtp::mtp_of(&draft, &target, &read.spec)
        .unwrap_or_else(|e| panic!("mtp_of {MOPD_MTP}: {e}"));
    let DraftSpec::Mtp(m) = spec else {
        panic!("{MOPD_MTP} read as a block draft")
    };
    let Mixer::Gqa(g) = &m.layer.mixer else {
        panic!("the MTP layer is {:?}, not GQA", m.layer.mixer)
    };
    let MtpSource::File {
        first_shard,
        bytes,
        borrows,
    } = &m.source
    else {
        panic!("the sidecar is a file source")
    };
    println!(
        "{MOPD_MTP}: layer {} hidden {} vocab {} gqa {}/{} x {} v {} window {:?} sinks {} \
         tensor bytes {bytes} borrows {borrows:?}",
        m.index, m.hidden, m.vocab, g.heads, g.kv_heads, g.head_dim, g.value_dim, g.window, g.sinks
    );
    assert_eq!(m.index, 48);
    assert_eq!((m.hidden, m.vocab), (4096, 152_576));
    assert_eq!(m.input, MtpInput::PreNorm);
    assert_eq!(m.head_norm, MtpHeadNorm::Rms);
    assert!(m.hc.is_none(), "{:?}", m.hc);
    assert_eq!(
        (g.heads, g.kv_heads, g.head_dim, g.value_dim),
        (64, 8, 192, 128)
    );
    assert_eq!((g.window, g.sinks), (Some(128), true));
    assert!(
        matches!(m.layer.ffn, Ffn::Dense { ff: 16_384, .. }),
        "{:?}",
        m.layer.ffn
    );
    assert!(
        first_shard.ends_with("mtp-MiMo-V2.6-Flash-MOPD-Q8_0.gguf"),
        "{first_shard:?}"
    );
    assert!(
        !borrows.embedding && !borrows.head,
        "the sidecar carries both matrices: {borrows:?}"
    );
}

/// The RL checkpoint's sidecar beside the MOPD target is refused on
/// `general.name`, naming both checkpoints, and by the table's rule before it
/// is fetched; the stub holds the real header, so nothing else differs.
#[test]
#[ignore = "needs the MiMo-V2.6-Flash-MOPD shards and the RL sidecar's census stub on the box (just gate-mimo2-meta)"]
fn hw_mimo2_mtp_sidecar_of_the_other_checkpoint_is_refused() {
    let target = Split::open(MOPD).unwrap_or_else(|e| panic!("open {MOPD}: {e}"));
    let read = model::arch::spec(&target).unwrap_or_else(|e| panic!("spec of {MOPD}: {e}"));
    let stub = Split::open(RL_MTP_STUB).unwrap_or_else(|e| panic!("open {RL_MTP_STUB}: {e}"));
    let err = model::arch::mimo2::mtp::mtp_of(&stub, &target, &read.spec)
        .expect_err("the RL sidecar beside the MOPD target is refused")
        .to_string();
    println!("{err}");
    assert!(err.starts_with("metadata general.name:"), "{err}");
    assert!(
        err.contains("MiMo-V2.6-Flash-MOPD") && err.contains("MiMo-V2.6-Flash-RL"),
        "{err}"
    );
    let rule = model::arch::draft::draft_usable(&target, Some("MiMo-V2.6-Flash-RL"))
        .expect_err("the table refuses the RL file for the MOPD set")
        .to_string();
    assert!(
        rule.contains("MiMo-V2.6-Flash-MOPD") && rule.contains("MiMo-V2.6-Flash-RL"),
        "{rule}"
    );
    model::arch::draft::draft_usable(&target, Some("MiMo-V2.6-Flash-MOPD"))
        .unwrap_or_else(|e| panic!("the MOPD file for the MOPD set: {e}"));
}

/// `shard`'s metadata rewritten: the `split.*` keys dropped, so the copy is
/// a one-shard file of no tensor, and the array `key` left with its first
/// `len` values.
fn doctored_kv(shard: &Gguf, key: &str, len: usize) -> Vec<u8> {
    let kv: Vec<(String, Value)> = shard
        .iter_kv()
        .filter(|(k, _)| !k.starts_with("split."))
        .map(|(k, v)| {
            if k != key {
                return (k.to_string(), v.clone());
            }
            let Value::Array(items) = v else {
                panic!("{key} is not an array");
            };
            assert!(items.len() > len, "{key} holds {} values", items.len());
            (
                k.to_string(),
                Value::Array(items.iter().take(len).cloned().collect()),
            )
        })
        .collect();
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    b.extend_from_slice(&(kv.len() as u64).to_le_bytes());
    for (k, v) in &kv {
        put_str(&mut b, k);
        put_value(&mut b, v);
    }
    b
}

/// One metadata value as the file's bytes: the type tag then the value, an
/// array's element tag before its count.
fn put_value(b: &mut Vec<u8>, v: &Value) {
    b.extend_from_slice(&tag_of(v).to_le_bytes());
    put_raw(b, v);
}

/// The value's bytes without its type tag.
fn put_raw(b: &mut Vec<u8>, v: &Value) {
    match v {
        Value::U8(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::I8(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::U16(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::I16(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::U32(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::I32(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::F32(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::Bool(x) => b.push(u8::from(*x)),
        Value::String(s) => put_str(b, s),
        Value::Array(items) => {
            let Some(first) = items.first() else {
                panic!("an empty array has no element tag to write");
            };
            b.extend_from_slice(&tag_of(first).to_le_bytes());
            b.extend_from_slice(&(items.len() as u64).to_le_bytes());
            items.iter().for_each(|x| put_raw(b, x));
        }
        Value::U64(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::I64(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::F64(x) => b.extend_from_slice(&x.to_le_bytes()),
    }
}

/// The type tag a value is written with.
fn tag_of(v: &Value) -> u32 {
    match v {
        Value::U8(_) => 0,
        Value::I8(_) => 1,
        Value::U16(_) => 2,
        Value::I16(_) => 3,
        Value::U32(_) => 4,
        Value::I32(_) => 5,
        Value::F32(_) => 6,
        Value::Bool(_) => 7,
        Value::String(_) => 8,
        Value::Array(_) => 9,
        Value::U64(_) => 10,
        Value::I64(_) => 11,
        Value::F64(_) => 12,
    }
}

/// One length-prefixed string.
fn put_str(b: &mut Vec<u8>, s: &str) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
}
