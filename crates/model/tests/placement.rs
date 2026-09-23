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
//! `hw_`: needs the V4.1 shards on the box (`just gate-placement`);
//! `BLOOMERY_V41_MODEL` names another first shard. The plan inputs are this
//! machine's figures in `model::placement::workstation`, which the GPU load
//! gate (`gate_load_v41`) plans from too.

use std::fmt::Write as _;
use std::ops::Range;

use gguf::{GgmlType, Split, Value};
use model::arch::deepseek41::{kv::KvLayout, roles};
use model::placement::{
    self, CardFormat, Device, Format, Machine, ModelTensors, Plan, Role, workstation,
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

// PIN(2026-09-23): design §5 (a)'s A6000 line with the allocator's rounding as a card term (its rule matched gate ②'s loads to the byte); n_l 62–63 on layers 2–39.
const A_A6000: CardPin = CardPin {
    card: "A6000",
    dense: 8_576_674_240,
    expert_bytes: 40_020_664_320,
    experts: 2_386,
    rounding: 553_612_864,
    kv: 110_129_152,
    headroom: 1_087_344_640,
    eligible: 2..40,
    n_l: (62, 63),
};
// PIN(2026-09-23): design §5 (a)'s host line, with the experts the A6000 no longer keeps once its rounding is counted.
const A_HOST: HostPin = HostPin {
    expert_bytes: 218_746_920_960,
    table_bytes: 1_323_827_200,
    headroom: 39_010_656_256,
};
// PIN(2026-09-23): design §5 (b)'s A6000 line with the allocator's rounding as a card term (its rule matched gate ②'s loads to the byte); n_l 148–149 on layers 2–19.
const B_A6000: CardPin = CardPin {
    card: "A6000",
    dense: 4_190_828_000,
    expert_bytes: 44_750_684_160,
    experts: 2_668,
    rounding: 257_673_760,
    kv: 65_560_576,
    headroom: 1_083_678_720,
    eligible: 2..20,
    n_l: (148, 149),
};
// PIN(2026-09-23): design §5 (b)'s 3090 line with the allocator's rounding as a card term (its rule matched gate ②'s loads to the byte); n_l 56–57 on layers 20–39.
const B_3090: CardPin = CardPin {
    card: "3090",
    dense: 4_385_846_240,
    expert_bytes: 18_970_398_720,
    experts: 1_131,
    rounding: 268_172_320,
    kv: 44_568_576,
    headroom: 1_077_407_744,
    eligible: 20..40,
    n_l: (56, 57),
};
// PIN(2026-09-23): design §5 (b)'s host line (token_embd as in (a)), with the experts the cards no longer keep once their rounding is counted.
const B_HOST: HostPin = HostPin {
    expert_bytes: 195_046_502_400,
    table_bytes: 1_323_827_200,
    headroom: 62_711_074_816,
};
// PIN(2026-09-23): design §5 (a), `engram_embd` ×2 on NVMe — the same in (b).
const NVME: u64 = 208_902_215_200;
// PIN(2026-09-23): design §3, the whole model placed once.
const TENSORS: usize = 1_046;
// PIN(2026-09-23): design §1, q8_0 outside the engram tables: tensors and plane bytes, all on cards.
const Q8_0_CARD: (usize, u64) = (330, 7_691_304_960);
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
    let model = roles::classify(&split).unwrap_or_else(|e| panic!("classify {path}: {e}"));
    let kv = KvLayout::from_file(&split).unwrap_or_else(|e| panic!("KV layout of {path}: {e}"));
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
                    .map_or("-".to_string(), |e| format!("{}..{}", e.start, e.end)),
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
