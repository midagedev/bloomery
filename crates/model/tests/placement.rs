//! Placement gate (`docs/v41-placement.md` §7 ①): V4.1-Flash's table on this
//! machine from the nine shard headers alone — no tensor bytes, no GPU, no
//! lease, seconds. One test per plan of design §5: (a) the A6000 and DDR4;
//! (b) the A6000 on layers 0–19, the 3090 on 20–39 with the head. Each prints
//! the per-device and per-layer summaries, then fails with the whole list of
//! what is wrong: every `Plan::violations` entry and every pin the plan misses
//! (device totals with the allocator's rounding, expert counts and n_l, the
//! per-type formats, the per-token read). `BLOOMERY_PLACEMENT_TABLE=1` also
//! prints the per-tensor table.
//!
//! A third test pins what a hot list changes: which experts a card keeps,
//! never how many or at what bytes. A fourth pins what a card budget is: the
//! card's usable bytes, so the A6000 under the 3090's budget plans the gate
//! placement's experts, and a budget below the dense floor is refused.
//!
//! Two synthetic contracts need no file and run in the fast loop: a tensor
//! its role puts on a card, of a type with no card format, is refused by
//! name, while a routed stack of such a type stays on the host; a dense FFN
//! sits on its layer's card and a never-loaded tensor is placed nowhere,
//! even with a layer past the model's.
//!
//! `hw_`: needs the V4.1 shards on the box (`just gate-placement`);
//! `BLOOMERY_V41_MODEL` names another first shard. The plan inputs are this
//! machine's figures in `model::placement::workstation`, which the GPU load
//! gate (`gate_load_v41`) plans from too.

use std::fmt::Write as _;
use std::ops::Range;

use gguf::{GgmlType, Split, Value};
use model::arch::deepseek41::{hparams::Hparams, kv::KvLayout, roles};
use model::placement::{
    self, CardFormat, CardTotals, Device, Format, HotList, KvBytes, Machine, ModelTensor,
    ModelTensors, PlacementError, Plan, Role, workstation,
};

/// One card's pinned totals, and the n_l band on the layers that can hold experts.
struct CardPin {
    card: &'static str,
    dense: u64,
    expert_bytes: u64,
    experts: u64,
    rounding: u64,
    kv: u64,
    headroom: i128,
    eligible: Range<usize>,
    n_l: (u64, u64),
}

struct HostPin {
    expert_bytes: u64,
    table_bytes: u64,
    headroom: i128,
}

// PIN(2026-09-23): design §5 (a)'s A6000 line after the q8_0 scale plane went f16 (planes = file bytes, 427,294,720 B less dense), the allocator's rounding a card term; n_l 63–64 on layers 2–39; KV 4,096 B less and headroom as much more since the ratio-1 source (layer 20, a group of one row) keeps no pooling state.
const A_A6000: CardPin = CardPin {
    card: "A6000",
    dense: 8_149_379_520,
    expert_bytes: 40_490_311_680,
    experts: 2_414,
    rounding: 515_454_528,
    kv: 110_125_056,
    headroom: 1_083_154_432,
    eligible: 2..40,
    n_l: (63, 64),
};
// PIN(2026-09-23): design §5 (a)'s host line after the f16 scale plane: every expert the A6000 does not keep, its rounding counted.
const A_HOST: HostPin = HostPin {
    expert_bytes: 218_277_273_600,
    table_bytes: 1_323_827_200,
    headroom: 39_480_303_616,
};
// PIN(2026-09-23): design §5 (b)'s A6000 line after the q8_0 scale plane went f16, the allocator's rounding a card term; n_l 148–149 on layers 2–19.
const B_A6000: CardPin = CardPin {
    card: "A6000",
    dense: 3_967_677_920,
    expert_bytes: 44_951_961_600,
    experts: 2_680,
    rounding: 277_449_248,
    kv: 65_560_576,
    headroom: 1_085_775_872,
    eligible: 2..20,
    n_l: (148, 149),
};
// PIN(2026-09-23): design §5 (b)'s 3090 line after the q8_0 scale plane went f16, the allocator's rounding a card term; n_l 57–58 on layers 20–39; KV 4,096 B less and headroom as much more since the ratio-1 source (layer 20, a group of one row) keeps no pooling state.
const B_3090: CardPin = CardPin {
    card: "3090",
    dense: 4_181_701_600,
    expert_bytes: 19_188_449_280,
    experts: 1_144,
    rounding: 250_072_096,
    kv: 44_564_480,
    headroom: 1_081_606_144,
    eligible: 20..40,
    n_l: (57, 58),
};
// PIN(2026-09-23): design §5 (b)'s host line (token_embd as in (a)) after the f16 scale plane: every expert the cards do not keep, their rounding counted.
const B_HOST: HostPin = HostPin {
    expert_bytes: 194_627_174_400,
    table_bytes: 1_323_827_200,
    headroom: 63_130_402_816,
};
// PIN(2026-09-23): design §5 (a), `engram_embd` ×2 on NVMe — the same in (b).
const NVME: u64 = 208_902_215_200;
// PIN(2026-09-23): design §3, the whole model placed once.
const TENSORS: usize = 1_046;
// PIN(2026-09-23): design §1, q8_0 outside the engram tables: tensors and plane bytes, all on cards — the file's bytes, since the scale plane keeps each block's f16 bits.
const Q8_0_CARD: (usize, u64) = (330, 7_264_010_240);
// PIN(2026-09-23): design §1, bf16 decoded to f32 on the cards: the routers, and engram_{k,q}.
const BF16_ROUTERS: u64 = 314_572_800;
const BF16_ENGRAM: u64 = 327_680;
// PIN(2026-09-23): design §3, bytes one token reads in the file's format: all, and dense.
const READ_TOTAL: u64 = 12_035_196_096;
const READ_DENSE: u64 = 7_991_939_520;

/// `v` with thousands separators.
fn n(v: impl Into<i128>) -> String {
    let v: i128 = v.into();
    let digits = v.unsigned_abs().to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3 + 1);
    if v < 0 {
        out.push('-');
    }
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn open() -> (Split, ModelTensors, KvLayout) {
    let path = workstation::model_v41();
    let split = Split::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let hp = Hparams::read(&split).unwrap_or_else(|e| panic!("hyperparameters of {path}: {e}"));
    let model = roles::classify(&split, &hp).unwrap_or_else(|e| panic!("classify {path}: {e}"));
    let kv =
        KvLayout::from_file(&split, &hp).unwrap_or_else(|e| panic!("KV layout of {path}: {e}"));
    (split, model, kv)
}

/// A metadata value as the file states it; an array longer than 48 shows its
/// length and first 8 items.
fn show(v: &Value) -> String {
    let list = |items: &[Value]| items.iter().map(show).collect::<Vec<_>>().join(", ");
    match v {
        Value::Array(items) if items.len() > 48 => {
            format!("[{} items: {}, …]", items.len(), list(&items[..8]))
        }
        Value::Array(items) => format!("[{}]", list(items)),
        Value::String(s) => format!("{s:?}"),
        Value::U8(x) => x.to_string(),
        Value::I8(x) => x.to_string(),
        Value::U16(x) => x.to_string(),
        Value::I16(x) => x.to_string(),
        Value::U32(x) => x.to_string(),
        Value::I32(x) => x.to_string(),
        Value::U64(x) => x.to_string(),
        Value::I64(x) => x.to_string(),
        Value::F32(x) => format!("{x:?}"),
        Value::F64(x) => format!("{x:?}"),
        Value::Bool(x) => x.to_string(),
    }
}

/// The file's own metadata under its architecture's prefix — the keys the
/// design could not read (§9).
fn metadata(out: &mut String, split: &Split) {
    let prefix = format!("{}.", split.architecture().unwrap_or("?"));
    let keys: Vec<(&str, &Value)> = split
        .iter_kv()
        .filter(|(k, _)| k.starts_with(&prefix))
        .collect();
    let _ = writeln!(out, "{prefix}* metadata, {} keys:", keys.len());
    for (k, v) in keys {
        let _ = writeln!(out, "  {k} = {}", show(v));
    }
}

fn summary(out: &mut String, title: &str, plan: &Plan<'_>, kv: &KvLayout) {
    let _ = writeln!(out, "{title}, ctx_max {}", n(plan.ctx_max));
    let sources: Vec<String> = kv.sources().map(|(l, r)| format!("{l}(r={r})")).collect();
    let _ = writeln!(out, "  compressed-stream sources: {}", sources.join(" "));
    for (card, t) in plan.machine.cards.iter().zip(&plan.cards) {
        let _ = writeln!(
            out,
            "  {} layers {:?}{}: dense {}  experts {} ({})  rounding {}  KV {}  scratch {}  context {}  headroom {}",
            card.name,
            card.layers,
            if card.head { " +head" } else { "" },
            n(t.dense_bytes),
            n(t.expert_bytes),
            n(t.experts),
            n(t.rounding_bytes),
            n(t.kv_bytes),
            n(t.scratch_bytes),
            n(t.context_bytes),
            n(t.headroom_bytes)
        );
    }
    let h = &plan.host;
    let _ = writeln!(
        out,
        "  host: experts {} ({})  tables {}  reserves {}  headroom {}",
        n(h.expert_bytes),
        n(h.experts),
        n(h.table_bytes),
        n(h.reserve_bytes),
        n(h.headroom_bytes)
    );
    let _ = writeln!(out, "  nvme: {}", n(plan.nvme_bytes));
    for (i, chunk) in plan.n_l.chunks(10).enumerate() {
        let cells: Vec<String> = chunk
            .iter()
            .enumerate()
            .map(|(j, v)| format!("{:>2}:{v:>3}", i * 10 + j))
            .collect();
        let _ = writeln!(out, "  n_l {}", cells.join("  "));
    }
    let (total, dense) = reads(plan);
    let _ = writeln!(out, "  read/token: total {}  dense {}", n(total), n(dense));
    let _ = writeln!(out, "  by type (count, file, on cards, on host, on NVMe):");
    let mut types: Vec<GgmlType> = plan.model.tensors.iter().map(|t| t.ty).collect();
    types.sort();
    types.dedup();
    for ty in types {
        let (mut count, mut file, mut sums) = (0usize, 0u64, [0u64; 3]);
        for r in plan
            .rows
            .iter()
            .filter(|r| plan.model.tensors[r.tensor].ty == ty)
        {
            count += 1;
            file += plan.model.tensors[r.tensor].file_bytes;
            for s in &r.segments {
                match s.device {
                    Device::Card(_) => sums[0] += s.resident_bytes,
                    Device::Host => sums[1] += s.resident_bytes,
                    Device::Nvme => sums[2] += s.resident_bytes,
                    Device::Unused => {}
                }
            }
        }
        let _ = writeln!(
            out,
            "    {:5} {count:4}  {:>17}  {:>17}  {:>17}  {:>17}",
            ty.to_string(),
            n(file),
            n(sums[0]),
            n(sums[1]),
            n(sums[2])
        );
    }
}

/// Bytes one token reads: all, and dense — everything but the routed experts
/// and the engram table rows (`token_embd`'s one row counts as dense).
fn reads(plan: &Plan<'_>) -> (u64, u64) {
    let total = plan.rows.iter().map(|r| r.read_bytes).sum();
    let dense = plan
        .rows
        .iter()
        .filter(|r| {
            !matches!(
                plan.model.tensors[r.tensor].role,
                Role::RoutedExperts | Role::EngramTable
            )
        })
        .map(|r| r.read_bytes)
        .sum();
    (total, dense)
}

/// Design §7 ①'s columns, one line per segment.
fn table(out: &mut String, plan: &Plan<'_>) {
    let _ = writeln!(
        out,
        "tensor\tlayer\trole\ttype\tfile_bytes\tdevice\tformat\tresident_bytes\texperts\tread_bytes\tstage"
    );
    for r in &plan.rows {
        let t = &plan.model.tensors[r.tensor];
        for s in &r.segments {
            let device = match s.device {
                Device::Card(c) => plan.machine.cards[c].name.clone(),
                Device::Host => "host".to_string(),
                Device::Nvme => "nvme".to_string(),
                Device::Unused => "-".to_string(),
            };
            let _ = writeln!(
                out,
                "{}\t{}\t{}\t{}\t{}\t{device}\t{}\t{}\t{}\t{}\t{}",
                t.name,
                t.layer.map_or("-".to_string(), |l| l.to_string()),
                t.role,
                t.ty,
                t.file_bytes,
                s.format,
                s.resident_bytes,
                s.experts
                    .as_ref()
                    .map_or("-".to_string(), ToString::to_string),
                r.read_bytes,
                plan.machine.cards[r.stage].name
            );
        }
    }
}

/// A mismatch line when `got` is not `want`.
fn pin(bad: &mut Vec<String>, what: &str, got: impl Into<i128>, want: impl Into<i128>) {
    let (got, want) = (got.into(), want.into());
    if got != want {
        let d = got - want;
        bad.push(format!(
            "{what}: plan {}, pin {} ({}{})",
            n(got),
            n(want),
            if d > 0 { "+" } else { "" },
            n(d)
        ));
    }
}

/// Resident bytes on cards of the rows `keep` selects, each segment in `format`.
fn on_cards(
    plan: &Plan<'_>,
    bad: &mut Vec<String>,
    format: CardFormat,
    keep: impl Fn(Role, GgmlType) -> bool,
) -> (usize, u64) {
    let (mut count, mut bytes) = (0, 0);
    for r in &plan.rows {
        let t = &plan.model.tensors[r.tensor];
        if !keep(t.role, t.ty) {
            continue;
        }
        count += 1;
        for s in &r.segments {
            if !matches!(s.device, Device::Card(_)) || s.format != Format::Card(format) {
                bad.push(format!(
                    "{}: {:?} as {}, not on a card as {format:?}",
                    t.name, s.device, s.format
                ));
            }
            bytes += s.resident_bytes;
        }
    }
    (count, bytes)
}

fn pins(plan: &Plan<'_>, cards: &[CardPin], host: &HostPin) -> Vec<String> {
    let mut bad = Vec::new();
    pin(
        &mut bad,
        "tensors placed",
        plan.rows.len() as u64,
        TENSORS as u64,
    );
    for ((p, card), t) in cards.iter().zip(&plan.machine.cards).zip(&plan.cards) {
        let c = p.card;
        assert_eq!(card.name, c, "pins are in stage order");
        pin(&mut bad, &format!("{c} dense"), t.dense_bytes, p.dense);
        pin(
            &mut bad,
            &format!("{c} expert bytes"),
            t.expert_bytes,
            p.expert_bytes,
        );
        pin(&mut bad, &format!("{c} experts"), t.experts, p.experts);
        pin(
            &mut bad,
            &format!("{c} allocator rounding"),
            t.rounding_bytes,
            p.rounding,
        );
        pin(&mut bad, &format!("{c} KV"), t.kv_bytes, p.kv);
        pin(
            &mut bad,
            &format!("{c} headroom"),
            t.headroom_bytes,
            p.headroom,
        );
        for l in card.layers.clone() {
            let band = if p.eligible.contains(&l) {
                p.n_l
            } else {
                (0, 0)
            };
            if !(band.0..=band.1).contains(&plan.n_l[l]) {
                bad.push(format!(
                    "{c} layer {l}: n_l {}, the pinned band is {}–{}",
                    plan.n_l[l], band.0, band.1
                ));
            }
        }
    }
    let h = &plan.host;
    pin(
        &mut bad,
        "host expert bytes",
        h.expert_bytes,
        host.expert_bytes,
    );
    pin(
        &mut bad,
        "host tables (token_embd)",
        h.table_bytes,
        host.table_bytes,
    );
    pin(&mut bad, "host headroom", h.headroom_bytes, host.headroom);
    pin(&mut bad, "nvme (engram_embd)", plan.nvme_bytes, NVME);

    let (count, bytes) = on_cards(plan, &mut bad, CardFormat::Q8_0Planes, |role, ty| {
        ty == GgmlType::Q8_0 && role != Role::EngramTable
    });
    pin(
        &mut bad,
        "q8_0 outside the engram tables: tensors",
        count as u64,
        Q8_0_CARD.0 as u64,
    );
    pin(
        &mut bad,
        "q8_0 outside the engram tables: planes",
        bytes,
        Q8_0_CARD.1,
    );
    let (_, bytes) = on_cards(plan, &mut bad, CardFormat::Bf16AsF32, |role, ty| {
        ty == GgmlType::BF16 && role == Role::Router
    });
    pin(&mut bad, "bf16 routers as f32", bytes, BF16_ROUTERS);
    let (_, bytes) = on_cards(plan, &mut bad, CardFormat::Bf16AsF32, |role, ty| {
        ty == GgmlType::BF16 && role == Role::EngramDense
    });
    pin(&mut bad, "bf16 engram_{k,q} as f32", bytes, BF16_ENGRAM);
    for r in &plan.rows {
        let t = &plan.model.tensors[r.tensor];
        let want = match t.ty {
            GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q6_K => CardFormat::KQuant,
            GgmlType::Q8_0 => CardFormat::Q8_0Planes,
            GgmlType::F32 => CardFormat::F32,
            GgmlType::BF16 => CardFormat::Bf16AsF32,
            _ => continue,
        };
        for s in r
            .segments
            .iter()
            .filter(|s| matches!(s.device, Device::Card(_)))
        {
            if s.format != Format::Card(want) {
                bad.push(format!(
                    "{}: {} on a card as {}, not {want:?}",
                    t.name, t.ty, s.format
                ));
            }
        }
        if t.role == Role::TokenEmbedding
            && !matches!(r.segments.as_slice(), [s] if s.device == Device::Host && s.format == Format::HostFile)
        {
            bad.push(format!(
                "{}: not one host segment in the file's format",
                t.name
            ));
        }
    }
    let (total, dense) = reads(plan);
    pin(&mut bad, "read/token total", total, READ_TOTAL);
    pin(&mut bad, "read/token dense", dense, READ_DENSE);
    bad
}

fn run(
    title: &str,
    mut out: String,
    model: &ModelTensors,
    machine: &Machine,
    kv: &KvLayout,
    cards: &[CardPin],
    host: &HostPin,
) {
    let plan = placement::plan(model, machine, workstation::CTX_MAX, kv)
        .unwrap_or_else(|e| panic!("{title}: {e}"));
    summary(&mut out, title, &plan, kv);
    if std::env::var("BLOOMERY_PLACEMENT_TABLE").is_ok_and(|v| v == "1") {
        table(&mut out, &plan);
    }
    let mut bad: Vec<String> = plan.violations().iter().map(ToString::to_string).collect();
    bad.extend(pins(&plan, cards, host));
    println!("{out}");
    assert!(
        bad.is_empty(),
        "{title}: {} failures\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}

#[test]
#[ignore = "hw: needs the V4.1 shards on the box"]
fn hw_placement_a6000_ddr4() {
    let (split, model, kv) = open();
    let mut out = String::new();
    metadata(&mut out, &split);
    let machine = workstation::plan_a(model.layers);
    run(
        "plan (a) A6000 + DDR4",
        out,
        &model,
        &machine,
        &kv,
        &[A_A6000],
        &A_HOST,
    );
}

#[test]
#[ignore = "hw: needs the V4.1 shards on the box"]
fn hw_placement_3090_a6000_cut20() {
    let (_, model, kv) = open();
    let machine = workstation::plan_b(model.layers);
    run(
        "plan (b) A6000 0-19 + 3090 20-39 +head + DDR4",
        String::new(),
        &model,
        &machine,
        &kv,
        &[B_A6000, B_3090],
        &B_HOST,
    );
}

/// A hot list moves which experts a card keeps, not how many: plan (a) with
/// a scattered synthetic list — layer `l`'s rank `r` is expert
/// `(7r + l) mod n_expert`, a permutation, so a card's experts fall in many
/// runs — keeps the id-prefix plan's `n_l` and every device total (dense,
/// expert bytes and count, rounding, headroom; host experts and bytes),
/// breaks no invariant, and each routed stack's card segment is its layer's
/// first `n_l` ranked ids, its host segment the rest.
#[test]
#[ignore = "hw: needs the V4.1 shards on the box"]
fn hw_placement_hot_list_keeps_the_counts() {
    let (_, model, kv) = open();
    let machine = workstation::plan_a(model.layers);
    let e = model.experts;
    let mut text = format!("# n_expert\t{e}\n# order\trank\n");
    for l in 0..model.layers as u64 {
        let ids: Vec<String> = (0..e).map(|r| ((7 * r + l) % e).to_string()).collect();
        let _ = writeln!(text, "{l}\t{}", ids.join(","));
    }
    let hot = HotList::parse("synthetic", &text).expect("the synthetic list parses");
    let ctx = workstation::CTX_MAX;
    let prefix = placement::plan_with(&model, &machine, ctx, &kv, None, None).expect("prefix plan");
    let listed =
        placement::plan_with(&model, &machine, ctx, &kv, Some(&hot), None).expect("list plan");
    let mut bad: Vec<String> = listed
        .violations()
        .iter()
        .map(ToString::to_string)
        .collect();
    if listed.n_l != prefix.n_l {
        bad.push(format!(
            "n_l {:?}, the prefix plan's {:?}",
            listed.n_l, prefix.n_l
        ));
    }
    for (c, (a, b)) in listed.cards.iter().zip(&prefix.cards).enumerate() {
        let got = (
            a.dense_bytes,
            a.expert_bytes,
            a.experts,
            a.rounding_bytes,
            a.headroom_bytes,
        );
        let want = (
            b.dense_bytes,
            b.expert_bytes,
            b.experts,
            b.rounding_bytes,
            b.headroom_bytes,
        );
        if got != want {
            bad.push(format!(
                "card {c}: (dense, experts B, experts, rounding, headroom) {got:?}, prefix {want:?}"
            ));
        }
    }
    let (h, p) = (&listed.host, &prefix.host);
    if (h.expert_bytes, h.experts, h.headroom_bytes)
        != (p.expert_bytes, p.experts, p.headroom_bytes)
    {
        bad.push(format!(
            "host (expert B, experts, headroom) ({}, {}, {}), prefix ({}, {}, {})",
            h.expert_bytes,
            h.experts,
            h.headroom_bytes,
            p.expert_bytes,
            p.experts,
            p.headroom_bytes
        ));
    }
    let (mut stacks, mut runs) = (0usize, 0usize);
    for r in &listed.rows {
        let t = &listed.model.tensors[r.tensor];
        let (Role::RoutedExperts, Some(l)) = (t.role, t.layer) else {
            continue;
        };
        stacks += 1;
        let n = usize::try_from(listed.n_l[l]).expect("n_l fits usize");
        let mut want_card: Vec<u32> = hot.ranked(l)[..n].to_vec();
        want_card.sort_unstable();
        let mut card = Vec::new();
        let mut host = Vec::new();
        for s in &r.segments {
            let ids = s.experts.as_ref().map_or(&[][..], |e| e.ids());
            match s.device {
                Device::Card(_) => {
                    card.extend_from_slice(ids);
                    runs += s.experts.as_ref().map_or(0, |e| e.runs().len());
                }
                Device::Host => host.extend_from_slice(ids),
                _ => bad.push(format!("{}: a segment on {:?}", t.name, s.device)),
            }
        }
        let mut want_host: Vec<u32> = (0..e as u32)
            .filter(|id| want_card.binary_search(id).is_err())
            .collect();
        want_host.sort_unstable();
        if card != want_card || host != want_host {
            bad.push(format!(
                "{}: card {} ids, host {} ids, the list's {n} and the rest",
                t.name,
                card.len(),
                host.len()
            ));
        }
    }
    println!(
        "hot list plan (a): {stacks} routed stacks, their card segments in {runs} runs; n_l, card \
         and host totals as the prefix plan's; {} failures",
        bad.len()
    );
    assert!(
        bad.is_empty(),
        "hot list plan: {} failures\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}

/// A card's totals the budget must reproduce: dense, expert bytes and count,
/// rounding, KV, headroom.
fn card_line(t: &CardTotals) -> (u64, u64, u64, u64, u64, i128) {
    (
        t.dense_bytes,
        t.expert_bytes,
        t.experts,
        t.rounding_bytes,
        t.kv_bytes,
        t.headroom_bytes,
    )
}

/// A card budget is the card's usable bytes and nothing else: plan (a) under
/// the 3090's usable bytes is the gate placement slot for slot — `n_l` per
/// layer and every card total, the card's name aside; under the A6000's own
/// usable bytes it is plan (a) unbudgeted; under 38 GiB it keeps more experts
/// than the first and fewer than the second, and breaks no invariant. A
/// budget below the card's floor (dense granules + KV + context + scratch +
/// margin) is refused with that floor; at the floor it plans, keeping no
/// expert on the card, and one byte less is refused.
#[test]
#[ignore = "hw: needs the V4.1 shards on the box"]
fn hw_placement_card_budget_is_usable_bytes() {
    let (_, model, kv) = open();
    let ctx = workstation::CTX_MAX;
    let a = workstation::plan_a(model.layers);
    let gate = workstation::plan_gate(model.layers);
    let with = |budget: Option<u64>| placement::plan_with(&model, &a, ctx, &kv, None, budget);
    let mut bad = Vec::new();

    let gate_plan = placement::plan_with(&model, &gate, ctx, &kv, None, None).expect("gate plan");
    let b3090 = workstation::RTX_3090.usable_bytes();
    let as_3090 = with(Some(b3090)).expect("plan (a) under the 3090's budget");
    if as_3090.n_l != gate_plan.n_l {
        bad.push(format!(
            "n_l under budget {b3090}: {:?}, the gate placement's {:?}",
            as_3090.n_l, gate_plan.n_l
        ));
    }
    if card_line(&as_3090.cards[0]) != card_line(&gate_plan.cards[0]) {
        bad.push(format!(
            "card under budget {b3090}: (dense, experts B, experts, rounding, KV, headroom) {:?}, \
             the gate placement's {:?}",
            card_line(&as_3090.cards[0]),
            card_line(&gate_plan.cards[0])
        ));
    }

    let unbudgeted = with(None).expect("plan (a)");
    let b_a6000 = workstation::A6000.usable_bytes();
    let as_a6000 = with(Some(b_a6000)).expect("plan (a) under its own budget");
    if as_a6000.n_l != unbudgeted.n_l
        || card_line(&as_a6000.cards[0]) != card_line(&unbudgeted.cards[0])
    {
        bad.push(format!(
            "plan (a) under its own usable {b_a6000} B: n_l {:?} card {:?}, unbudgeted n_l {:?} card {:?}",
            as_a6000.n_l,
            card_line(&as_a6000.cards[0]),
            unbudgeted.n_l,
            card_line(&unbudgeted.cards[0])
        ));
    }

    let b38 = 38u64 << 30;
    let mid = with(Some(b38)).expect("plan (a) under 38 GiB");
    let (lo, hi) = (gate_plan.cards[0].experts, unbudgeted.cards[0].experts);
    let got = mid.cards[0].experts;
    if !(lo < got && got < hi) {
        bad.push(format!(
            "38 GiB keeps {got} experts, not strictly between the gate's {lo} and plan (a)'s {hi}"
        ));
    }
    bad.extend(mid.violations().iter().map(|v| format!("38 GiB: {v}")));
    let held: Vec<u64> = mid.n_l.iter().copied().filter(|&n| n > 0).collect();
    println!("38 GiB n_l {:?}", mid.n_l);

    let floor = match with(Some(1 << 30)) {
        Err(PlacementError::CardBudgetFloor {
            floor,
            dense,
            kv,
            context,
            scratch,
            margin,
            ..
        }) => {
            if floor != dense + kv + context + scratch + margin {
                bad.push(format!(
                    "floor {floor} is not dense {dense} + KV {kv} + context {context} + scratch \
                     {scratch} + margin {margin}"
                ));
            }
            floor
        }
        other => panic!(
            "a 1 GiB budget: {:?}, not the floor refusal",
            other.map(|p| p.n_l)
        ),
    };
    match with(Some(floor)) {
        Ok(p) => {
            if p.n_l.iter().any(|&n| n > 0) {
                bad.push(format!(
                    "at the floor {floor} B the card keeps experts: {:?}",
                    p.n_l
                ));
            }
            bad.extend(
                p.violations()
                    .iter()
                    .filter(|v| matches!(v, placement::Violation::CardOver { .. }))
                    .map(|v| format!("at the floor: {v}")),
            );
        }
        Err(e) => bad.push(format!("at the floor {floor} B: {e}")),
    }
    match with(Some(floor - 1)) {
        Err(e @ PlacementError::CardBudgetFloor { .. }) => println!("one byte below: {e}"),
        other => bad.push(format!(
            "one byte below the floor {floor}: {:?}, not the floor refusal",
            other.map(|p| p.n_l)
        )),
    }

    println!(
        "card budget on plan (a): {b3090} B (3090) keeps {} experts, the gate's {}; {b38} B (38 GiB) \
         keeps {got}, n_l {}..{} on {} layers; {b_a6000} B (A6000) keeps {}, unbudgeted {hi}; floor \
         {floor} B; {} failures",
        as_3090.cards[0].experts,
        gate_plan.cards[0].experts,
        held.iter().min().copied().unwrap_or(0),
        held.iter().max().copied().unwrap_or(0),
        held.len(),
        as_a6000.cards[0].experts,
        bad.len()
    );
    assert!(
        bad.is_empty(),
        "card budget: {} failures\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}

/// No KV cache: the synthetic models' cards hold only their tensors.
struct NoKv;

impl KvBytes for NoKv {
    fn layer_bytes(&self, _layer: usize, _ctx_max: u64) -> u64 {
        0
    }
}

/// A tensor of layer `layer` (or model-level) with `rows` rows of 256 values
/// of type `ty`, sized by ggml's block table.
fn synthetic(
    name: &str,
    layer: Option<usize>,
    role: Role,
    ty: GgmlType,
    rows: &[u64],
) -> ModelTensor {
    let mut dims = vec![256];
    dims.extend_from_slice(rows);
    let blocks = 256 / ty.blck_size().expect("a sized type");
    let file_bytes = ty.type_size().expect("a sized type") * blocks * rows.iter().product::<u64>();
    ModelTensor {
        name: name.to_string(),
        shard: 0,
        layer,
        role,
        ty,
        dims,
        file_bytes,
        gathered_rows: (role == Role::TokenEmbedding).then_some(1),
    }
}

/// One layer of 8 experts, 2 used, with the given tensors besides the
/// embedding and the head.
fn synthetic_model(mut layer: Vec<ModelTensor>) -> ModelTensors {
    let mut tensors = vec![synthetic(
        "token_embd.weight",
        None,
        Role::TokenEmbedding,
        GgmlType::Q8_0,
        &[16],
    )];
    tensors.append(&mut layer);
    tensors.push(synthetic(
        "output.weight",
        None,
        Role::Head,
        GgmlType::Q8_0,
        &[16],
    ));
    ModelTensors {
        tensors,
        layers: 1,
        experts: 8,
        experts_used: 2,
    }
}

/// A required card tensor without a card format is refused with its name,
/// type and role; a routed stack of the same type is not refused — the plan
/// keeps every expert of it on the host.
#[test]
fn card_tensor_without_card_format_is_refused_by_name() {
    let machine = workstation::plan_a(1);
    let q5k_attn = synthetic_model(vec![synthetic(
        "blk.0.attn_q_b.weight",
        Some(0),
        Role::Attention,
        GgmlType::Q5_K,
        &[4],
    )]);
    match placement::plan_with(&q5k_attn, &machine, 4096, &NoKv, None, None) {
        Err(e @ PlacementError::NoCardFormat { .. }) => {
            let PlacementError::NoCardFormat { name, ty, role } = &e else {
                unreachable!()
            };
            assert_eq!(
                (name.as_str(), *ty, *role),
                ("blk.0.attn_q_b.weight", GgmlType::Q5_K, Role::Attention)
            );
            let text = e.to_string();
            assert!(
                text.contains("blk.0.attn_q_b.weight") && text.contains("q5_K"),
                "{text}"
            );
        }
        Err(e) => panic!("refused with the wrong error: {e}"),
        Ok(_) => panic!("a q5_K attention tensor was placed on a card"),
    }

    let q5k_stack = synthetic_model(vec![synthetic(
        "blk.0.ffn_down_exps.weight",
        Some(0),
        Role::RoutedExperts,
        GgmlType::Q5_K,
        &[4, 8],
    )]);
    let plan = placement::plan_with(&q5k_stack, &machine, 4096, &NoKv, None, None)
        .expect("a q5_K routed stack plans, on the host");
    assert!(plan.violations().is_empty(), "{:?}", plan.violations());
    assert_eq!(plan.n_l, vec![0]);
    let row = &plan.rows[1];
    assert!(
        matches!(row.segments.as_slice(), [s] if s.device == Device::Host && s.format == Format::HostFile),
        "{:?}",
        row.segments
    );
}

/// A dense FFN is placed like attention, whole on its layer's card; a
/// never-loaded tensor has one segment nowhere, reads nothing, and may
/// carry a layer past the model's and a shape no card upload takes (16 rows
/// of one q6_K block do not split into the KQuant words).
#[test]
fn dense_ffn_on_card_and_unused_nowhere() {
    let machine = workstation::plan_a(1);
    let model = synthetic_model(vec![
        synthetic(
            "blk.0.ffn_up.weight",
            Some(0),
            Role::DenseFfn,
            GgmlType::Q4_K,
            &[4],
        ),
        synthetic(
            "blk.1.nextn.eh_proj.weight",
            Some(1),
            Role::Unused,
            GgmlType::Q6_K,
            &[16],
        ),
    ]);
    let plan = placement::plan_with(&model, &machine, 4096, &NoKv, None, None)
        .expect("a dense FFN and an MTP head plan");
    assert!(plan.violations().is_empty(), "{:?}", plan.violations());
    let dense = &plan.rows[1];
    assert!(
        matches!(dense.segments.as_slice(), [s] if s.device == Device::Card(0) && s.format == Format::Card(CardFormat::KQuant)),
        "{:?}",
        dense.segments
    );
    assert_eq!(dense.read_bytes, model.tensors[1].file_bytes);
    let unused = &plan.rows[2];
    assert!(
        matches!(unused.segments.as_slice(), [s] if s.device == Device::Unused && s.format == Format::Unused && s.resident_bytes == 0),
        "{:?}",
        unused.segments
    );
    assert_eq!(unused.read_bytes, 0);
}
