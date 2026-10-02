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
//! A (b′) test plans the A6000 as (a) with the 3090 as an expert tier, with
//! and without the DSpark draft's reserve (`BLOOMERY_DSPARK_MODEL`, whose
//! card bytes it pins too): the A6000's rows are plan (a)'s, and the tier
//! holds each layer's next ids, the per-layer totals even.
//!
//! A third test pins what a card budget is: the card's usable bytes, so the
//! A6000 under the 3090's budget plans the gate placement's experts, and a
//! budget below the dense floor is refused.
//!
//! Synthetic contracts need no file and run in the fast loop: a tensor
//! its role puts on a card, of a type with no card format, is refused by
//! name, while a routed stack of such a type stays on the host; a dense FFN
//! sits on its layer's card and a never-loaded tensor is placed nowhere,
//! even with a layer past the model's; the tier rule (the next ids, the
//! fewest-first fill, reserves), a tier that claims a stage refused by name,
//! and an expert on two cards refused by name.
//!
//! `hw_`: needs the V4.1 shards on the box (`just gate-placement`);
//! `BLOOMERY_V41_MODEL` names another first shard. The plan inputs are this
//! machine's figures in `model::placement::workstation`, which the GPU load
//! gate (`gate_load_v41`) plans from too.

use std::fmt::Write as _;
use std::num::NonZeroU64;
use std::ops::Range;

use gguf::{GgmlType, Split, Value};
use model::arch::deepseek41::{hparams::Hparams, kv::KvLayout, roles};
use model::arch::dspark::{self, DraftCardBytes};
use model::arch::glm5next::place::card_routed;
use model::placement::{
    self, Card, CardFormat, CardTotals, Device, ExpertList, Format, KvBytes, Machine, ModelTensor,
    ModelTensors, PlacementError, Plan, PlanLevers, Role, workstation,
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
    /// The cards' ring shadows, page-locked: `ctx_max` rows of the latent
    /// width in f16 per layer.
    shadow: u64,
    headroom: i128,
}

// PIN(2026-09-23): design §3, the whole model placed once.
const TENSORS: usize = 1_046;
// PIN(2026-09-23): design §1, bf16 decoded to f32 on the cards: the routers.
const BF16_ROUTERS: u64 = 314_572_800;

// PIN(2026-09-24): the public Q3_K_M file, plan (a)'s A6000 line, re-taken on top of the host ring shadow: attention and the shared experts in q3_K/q4_K instead of q8_0 leave 4,258,054,144 dense bytes fewer, and the card keeps 254 experts more (2,668), n_l 70–71.
const PUB_A_A6000: CardPin = CardPin {
    card: "A6000",
    dense: 3_891_325_376,
    expert_bytes: 44_750_684_160,
    experts: 2_668,
    rounding: 515_233_344,
    kv: 110_125_056,
    headroom: 1_081_057_280,
    eligible: 2..40,
    n_l: (70, 71),
};
// PIN(2026-09-24): the public file, plan (a)'s host line, re-taken on top of the host ring shadow: 254 experts fewer, token_embd in q3_K (284,416,000 B) instead of bf16, the same ring shadows.
const PUB_A_HOST: HostPin = HostPin {
    expert_bytes: 214_016_901_120,
    table_bytes: 284_416_000,
    shadow: 1_342_177_280,
    headroom: 43_437_910_016,
};
// PIN(2026-09-24): the public file, plan (b)'s A6000 line, re-taken on top of the host ring shadow: 134 experts more (2,814), n_l 156–157 on layers 2–19.
const PUB_B_A6000: CardPin = CardPin {
    card: "A6000",
    dense: 1_743_902_176,
    expert_bytes: 47_199_559_680,
    experts: 2_814,
    rounding: 255_724_064,
    kv: 65_560_576,
    headroom: 1_083_678_720,
    eligible: 2..20,
    n_l: (156, 157),
};
// PIN(2026-09-24): the public file, plan (b)'s 3090 line, re-taken on top of the host ring shadow: 121 experts more (1,265), n_l 63–64 on layers 20–39.
const PUB_B_3090: CardPin = CardPin {
    card: "3090",
    dense: 2_147_423_200,
    expert_bytes: 21_217_996_800,
    experts: 1_265,
    rounding: 258_997_280,
    kv: 44_564_480,
    headroom: 1_077_411_840,
    eligible: 20..40,
    n_l: (63, 64),
};
// PIN(2026-09-24): the public file, plan (b)'s host line, re-taken on top of the host ring shadow.
const PUB_B_HOST: HostPin = HostPin {
    expert_bytes: 190_350_028_800,
    table_bytes: 284_416_000,
    shadow: 1_342_177_280,
    headroom: 67_104_782_336,
};

/// One file's pins: the public file's are the consts above with `PUB_` and
/// the literals in [`PUBLIC`].
struct FilePins {
    a_cards: &'static [CardPin],
    a_host: &'static HostPin,
    b_cards: &'static [CardPin],
    b_host: &'static HostPin,
    nvme: u64,
    q8_0_card: (usize, u64),
    q5_k_card: (usize, u64),
    engram_gain: u64,
    read_total: u64,
    read_dense: u64,
    gate_experts: u64,
}

const PUBLIC: FilePins = FilePins {
    a_cards: &[PUB_A_A6000],
    a_host: &PUB_A_HOST,
    b_cards: &[PUB_B_A6000, PUB_B_3090],
    b_host: &PUB_B_HOST,
    // PIN(2026-09-24): the public file's engram_embd ×2 in q3_K on NVMe.
    nvme: 84_482_513_500,
    // PIN(2026-09-24): the public file has no q8_0 tensor.
    q8_0_card: (0, 0),
    // PIN(2026-09-24): ffn_down_shexp of layers 0 and 1, q5_K, on the card for the dense gemv.
    q5_k_card: (2, 16_220_160),
    // PIN(2026-09-24): engram_{k,q} in q3_K, decoded to f32 on the cards: the mixed file's bf16 bytes, the same shapes.
    engram_gain: 327_680,
    // PIN(2026-09-24): the public file's bytes one token reads: all, and dense.
    read_total: 7_774_310_520,
    read_dense: 3_731_061_720,
    // PIN(2026-09-24): the public file's gate placement keeps 1,146 experts; re-taken on top of the host ring shadow.
    gate_experts: 1_146,
};

/// The pins of the file this run opens ([`workstation::model_v41`]): the
/// public file's. Any other file has none, and is refused by name rather
/// than held to another file's numbers.
fn file_pins() -> &'static FilePins {
    let path = workstation::model_v41();
    assert!(
        path == gguf::v41::PUBLIC,
        "{path}: the placement gate pins {} only",
        gguf::v41::PUBLIC
    );
    &PUBLIC
}

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
    for (card, t) in plan.machine.all_cards().zip(&plan.cards) {
        let _ = writeln!(
            out,
            "  {} layers {:?}{}: dense {}  experts {} ({})  rounding {}  KV {}  scratch {}  context {}  reserves {}  headroom {}",
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
            n(t.reserve_bytes),
            n(t.headroom_bytes)
        );
    }
    let h = &plan.host;
    let _ = writeln!(
        out,
        "  host: experts {} ({})  tables {}  ring shadows {}  reserves {}  headroom {}",
        n(h.expert_bytes),
        n(h.experts),
        n(h.table_bytes),
        n(h.shadow_bytes),
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
                Device::Card(c) => plan
                    .machine
                    .card(c)
                    .map_or(format!("#{c}"), |k| k.name.clone()),
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

fn pins(plan: &Plan<'_>, cards: &[CardPin], host: &HostPin, f: &FilePins) -> Vec<String> {
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
    pin(&mut bad, "host ring shadows", h.shadow_bytes, host.shadow);
    pin(&mut bad, "host headroom", h.headroom_bytes, host.headroom);
    pin(&mut bad, "nvme (engram_embd)", plan.nvme_bytes, f.nvme);

    let (count, bytes) = on_cards(plan, &mut bad, CardFormat::Q8_0Planes, |role, ty| {
        ty == GgmlType::Q8_0 && role != Role::EngramTable
    });
    pin(
        &mut bad,
        "q8_0 outside the engram tables: tensors",
        count as u64,
        f.q8_0_card.0 as u64,
    );
    pin(
        &mut bad,
        "q8_0 outside the engram tables: planes",
        bytes,
        f.q8_0_card.1,
    );
    // q5_K outside the routed stacks, which no card expert kernel reads.
    let (count, bytes) = on_cards(plan, &mut bad, CardFormat::KQuant, |role, ty| {
        ty == GgmlType::Q5_K && role != Role::RoutedExperts
    });
    pin(
        &mut bad,
        "q5_K on cards: tensors",
        count as u64,
        f.q5_k_card.0 as u64,
    );
    pin(&mut bad, "q5_K on cards: bytes", bytes, f.q5_k_card.1);
    let (_, bytes) = on_cards(plan, &mut bad, CardFormat::Bf16AsF32, |role, ty| {
        ty == GgmlType::BF16 && role == Role::Router
    });
    pin(&mut bad, "bf16 routers as f32", bytes, BF16_ROUTERS);
    let (_, bytes) = on_cards(plan, &mut bad, CardFormat::Bf16AsF32, |role, _| {
        role == Role::EngramGain
    });
    pin(&mut bad, "engram_{k,q} as f32", bytes, f.engram_gain);
    for r in &plan.rows {
        let t = &plan.model.tensors[r.tensor];
        let want = match t.ty {
            GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K
                if t.role == Role::EngramGain =>
            {
                CardFormat::Bf16AsF32
            }
            GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K => CardFormat::KQuant,
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
    pin(&mut bad, "read/token total", total, f.read_total);
    pin(&mut bad, "read/token dense", dense, f.read_dense);
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
    let levers = bloomery_levers::Levers::from_env().unwrap_or_else(|e| panic!("{title}: {e}"));
    let levers =
        placement::PlanLevers::from_levers(&levers).unwrap_or_else(|e| panic!("{title}: {e}"));
    let plan = placement::plan(model, machine, workstation::CTX_MAX, kv, &levers)
        .unwrap_or_else(|e| panic!("{title}: {e}"));
    summary(&mut out, title, &plan, kv);
    if std::env::var("BLOOMERY_PLACEMENT_TABLE").is_ok_and(|v| v == "1") {
        table(&mut out, &plan);
    }
    let mut bad: Vec<String> = plan.violations().iter().map(ToString::to_string).collect();
    bad.extend(pins(&plan, cards, host, file_pins()));
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
        file_pins().a_cards,
        file_pins().a_host,
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
        file_pins().b_cards,
        file_pins().b_host,
    );
}

/// An expert tier card's pinned side of a (b′) plan: its totals, and the
/// band of its per-layer counts on the layers that can hold experts.
struct TierPin {
    experts: u64,
    expert_bytes: u64,
    rounding: u64,
    reserve: u64,
    headroom: i128,
    n_l: (u64, u64),
}

// PIN(2026-09-28): the public file, plan (b′) with no draft: the 3090 as expert tier. The gate read the value derived on paper.
// PIN(2026-09-28): re-pinned for the tier prompt batch's reserve, 152,511,496 B (workstation::tier_batch_bytes at n_embd 5120, ff 2304, 512 columns): past the 1,401-expert plan's slack it frees ⌈(152,511,496 − slack) / 16,773,120⌉ = 9 experts, 1,401 → 1,392; headroom −1,516,552 = −reserve + 9 experts + 36,864 rounding.
const PUB_BP_TIER: TierPin = TierPin {
    experts: 1_392,
    expert_bytes: 23_348_183_040,
    rounding: 165_085_184,
    reserve: 152_511_496,
    headroom: 1_080_613_880,
    n_l: (36, 37),
};
// PIN(2026-09-28): the public file, plan (b′) with the DSpark draft's reserve on the 3090. The gate read the value derived on paper.
// PIN(2026-09-28): re-pinned for the tier prompt batch's reserve beside the draft's, 152,511,496 B: 8 experts, 889 → 881, and 16,809,984 B of rounding free it; headroom −1,516,552.
const PUB_BP_TIER_DRAFT: TierPin = TierPin {
    experts: 881,
    expert_bytes: 14_777_118_720,
    rounding: 97_980_416,
    reserve: 8_782_291_976,
    headroom: 1_089_002_488,
    n_l: (23, 24),
};

// PIN(2026-09-28): the draft's card bytes from its header and the public file's head and mask row. The gate read the value derived on paper.
const DRAFT_BYTES: DraftCardBytes = DraftCardBytes {
    weights: 8_514_626_592,
    kv: 393_216,
    rounding: 47_651_808,
    scratch: dspark::SCRATCH_ALLOWANCE,
};

/// The draft file `BLOOMERY_DSPARK_MODEL` names — the recipe exports the
/// V4.1 profile's `DSPARK_MODEL` (`tools/ref/models/deepseek41.sh`), the one
/// owner of its path — refused by name when unset; another file misses the
/// [`DRAFT_BYTES`] pins.
fn draft_bytes(target: &Split) -> DraftCardBytes {
    let path = std::env::var("BLOOMERY_DSPARK_MODEL").unwrap_or_else(|e| {
        panic!("BLOOMERY_DSPARK_MODEL: {e} — `just gate-placement` exports the V4.1 profile's DSPARK_MODEL")
    });
    println!("DSpark draft: {path}");
    let draft = Split::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    dspark::card_bytes(&draft, target, workstation::GRANULE)
        .unwrap_or_else(|e| panic!("the draft's card bytes of {path}: {e}"))
}

/// The segments `plan` puts on card `c`, row by row: (tensor, format,
/// experts, resident bytes).
fn on_card(plan: &Plan<'_>, c: usize) -> Vec<(usize, Format, Option<String>, u64)> {
    plan.rows
        .iter()
        .flat_map(|r| {
            r.segments
                .iter()
                .filter(|s| s.device == Device::Card(c))
                .map(|s| {
                    (
                        r.tensor,
                        s.format,
                        s.experts.as_ref().map(ToString::to_string),
                        s.resident_bytes,
                    )
                })
        })
        .collect()
}

/// Plan (b′) against plan (a) and its tier pins: no violation; the A6000's
/// rows, `n_l` and totals exactly plan (a)'s; the tier holds per layer the
/// ids after the A6000's `n_l` in id order, the per-layer totals within one of each other, the
/// host exactly the rest; and the tier's totals and per-layer band pinned,
/// with no draft and with the draft's reserve (its card bytes pinned too).
fn bp_failures(
    title: &str,
    model: &ModelTensors,
    a: &Plan<'_>,
    bp: &Plan<'_>,
    want: &TierPin,
) -> Vec<String> {
    let mut bad: Vec<String> = bp
        .violations()
        .iter()
        .map(|v| format!("{title}: {v}"))
        .collect();
    if bp.n_l != a.n_l {
        bad.push(format!(
            "{title}: A6000 n_l {:?}, plan (a)'s {:?}",
            bp.n_l, a.n_l
        ));
    }
    if card_line(&bp.cards[0]) != card_line(&a.cards[0]) {
        bad.push(format!(
            "{title}: A6000 (dense, experts B, experts, rounding, KV, headroom) {:?}, plan (a)'s {:?}",
            card_line(&bp.cards[0]),
            card_line(&a.cards[0])
        ));
    }
    if on_card(bp, 0) != on_card(a, 0) {
        bad.push(format!(
            "{title}: the A6000's segments differ from plan (a)'s"
        ));
    }
    let tier = &bp.tier_n_l[0];
    let t = &bp.cards[1];
    let p = |bad: &mut Vec<String>, what: &str, got: i128, pinned: i128| {
        pin(bad, &format!("{title}: tier {what}"), got, pinned);
    };
    p(&mut bad, "dense", t.dense_bytes.into(), 0);
    p(&mut bad, "KV", t.kv_bytes.into(), 0);
    p(&mut bad, "experts", t.experts.into(), want.experts.into());
    p(
        &mut bad,
        "expert bytes",
        t.expert_bytes.into(),
        want.expert_bytes.into(),
    );
    p(
        &mut bad,
        "allocator rounding",
        t.rounding_bytes.into(),
        want.rounding.into(),
    );
    p(
        &mut bad,
        "reserves",
        t.reserve_bytes.into(),
        want.reserve.into(),
    );
    p(&mut bad, "headroom", t.headroom_bytes, want.headroom);
    let eligible: Vec<usize> = (0..model.layers).filter(|&l| a.n_l[l] > 0).collect();
    let held: Vec<u64> = eligible.iter().map(|&l| tier[l]).collect();
    let (lo, hi) = (
        held.iter().min().copied().unwrap_or(0),
        held.iter().max().copied().unwrap_or(0),
    );
    if (lo, hi) != want.n_l {
        bad.push(format!(
            "{title}: tier n_l {lo}–{hi} on the A6000's {} expert layers, the pinned band {}–{}",
            eligible.len(),
            want.n_l.0,
            want.n_l.1
        ));
    }
    for l in (0..model.layers).filter(|l| !eligible.contains(l)) {
        if tier[l] != 0 {
            bad.push(format!(
                "{title}: tier layer {l} holds {} experts, the A6000 none",
                tier[l]
            ));
        }
    }
    let totals: Vec<u64> = eligible.iter().map(|&l| a.n_l[l] + tier[l]).collect();
    let (tlo, thi) = (
        totals.iter().min().copied().unwrap_or(0),
        totals.iter().max().copied().unwrap_or(0),
    );
    if thi > tlo + 1 {
        bad.push(format!("{title}: per-layer totals {tlo}–{thi}, not even"));
    }
    let e = model.experts;
    for r in &bp.rows {
        let tensor = &bp.model.tensors[r.tensor];
        let (Role::RoutedExperts, Some(l)) = (tensor.role, tensor.layer) else {
            continue;
        };
        let (n, k) = (a.n_l[l], tier[l]);
        let want_tier: Vec<u32> = (n as u32..(n + k) as u32).collect();
        let mut got_tier = Vec::new();
        let mut host = 0u64;
        for s in &r.segments {
            let ids = s.experts.as_ref().map_or(&[][..], |x| x.ids());
            match s.device {
                Device::Card(1) => got_tier.extend_from_slice(ids),
                Device::Host => host += ids.len() as u64,
                _ => {}
            }
        }
        if got_tier != want_tier || host != e - n - k {
            bad.push(format!(
                "{title}: {}: tier {} ids, host {host}; want the ids {n}..{} ({} ids) and the rest",
                tensor.name,
                got_tier.len(),
                n + k,
                want_tier.len()
            ));
        }
    }
    let (ah, bh) = (&a.host, &bp.host);
    if (ah.experts - bh.experts, ah.expert_bytes - bh.expert_bytes) != (t.experts, t.expert_bytes) {
        bad.push(format!(
            "{title}: the host lost {} experts ({} B), the tier holds {} ({} B)",
            ah.experts - bh.experts,
            ah.expert_bytes - bh.expert_bytes,
            t.experts,
            t.expert_bytes
        ));
    }
    println!(
        "{title}: tier {} experts ({} B), rounding {}, reserves {}, headroom {}; tier n_l {lo}–{hi}, \
         per-layer totals {tlo}–{thi}; tier n_l {:?}",
        n(t.experts),
        n(t.expert_bytes),
        n(t.rounding_bytes),
        n(t.reserve_bytes),
        n(t.headroom_bytes),
        tier
    );
    bad
}

#[test]
#[ignore = "hw: needs the V4.1 shards and the DSpark draft on the box"]
fn hw_placement_bp() {
    let (split, model, kv) = open();
    let ctx = workstation::CTX_MAX;
    let a_machine = workstation::plan_a(model.layers);
    let a = placement::plan_with(&model, &a_machine, ctx, &kv, None).expect("plan (a)");
    let mut bad = Vec::new();

    let d = draft_bytes(&split);
    let hp = Hparams::read(&split).expect("the hyperparameters open() read");
    let batch = workstation::tier_batch_bytes(
        hp.n_embd as u64,
        hp.experts.ff as u64,
        hp.experts.n_used as u64,
        model::moe::UNION_MAX_COLS as u64,
    );
    println!(
        "tier prompt batch: staging {}  tile scratch {}  card {}  host {}",
        n(batch.staging),
        n(batch.scratch),
        n(batch.card()),
        n(batch.host)
    );
    println!(
        "DSpark draft on its card: weights {}  KV {}  rounding {}  scratch {}  total {}",
        n(d.weights),
        n(d.kv),
        n(d.rounding),
        n(d.scratch),
        n(d.total())
    );
    let dp = |bad: &mut Vec<String>, what: &str, got: u64, want: u64| {
        pin(bad, &format!("draft {what}"), got, want);
    };
    dp(&mut bad, "weights", d.weights, DRAFT_BYTES.weights);
    dp(&mut bad, "KV", d.kv, DRAFT_BYTES.kv);
    dp(&mut bad, "rounding", d.rounding, DRAFT_BYTES.rounding);
    dp(&mut bad, "scratch", d.scratch, DRAFT_BYTES.scratch);

    let mut out = String::new();
    for (title, draft, want) in [
        ("plan (b′) no draft", None, &PUB_BP_TIER),
        ("plan (b′) + draft", Some(d.total()), &PUB_BP_TIER_DRAFT),
    ] {
        let machine = workstation::plan_bp(model.layers, draft, batch);
        let bp = placement::plan_with(&model, &machine, ctx, &kv, None)
            .unwrap_or_else(|e| panic!("{title}: {e}"));
        summary(&mut out, title, &bp, &kv);
        bad.extend(bp_failures(title, &model, &a, &bp, want));
    }
    println!("{out}");
    println!(
        "plan (b′): derived 1,404 tier experts without the draft and 873 with it; {} failures",
        bad.len()
    );
    assert!(
        bad.is_empty(),
        "plan (b′): {} failures\n  {}",
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
    let with = |budget: Option<u64>| placement::plan_with(&model, &a, ctx, &kv, budget);
    let mut bad = Vec::new();

    let gate_plan = placement::plan_with(&model, &gate, ctx, &kv, None).expect("gate plan");
    pin(
        &mut bad,
        "gate placement experts",
        gate_plan.cards[0].experts,
        file_pins().gate_experts,
    );
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
            reserves,
            margin,
            ..
        }) => {
            if floor != dense + kv + context + scratch + reserves + margin {
                bad.push(format!(
                    "floor {floor} is not dense {dense} + KV {kv} + context {context} + scratch \
                     {scratch} + reserves {reserves} + margin {margin}"
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

/// A required card tensor without a card format (q2_K) is refused with its
/// name, type and role; a routed stack of a type no card expert kernel reads
/// is not refused — the plan keeps every expert of it on the host — whether
/// its type has no card format at all (q2_K) or only a dense one (q5_K).
#[test]
fn card_tensor_without_card_format_is_refused_by_name() {
    let machine = workstation::plan_a(1);
    let q2k_attn = synthetic_model(vec![synthetic(
        "blk.0.attn_q_b.weight",
        Some(0),
        Role::Attention,
        GgmlType::Q2_K,
        &[4],
    )]);
    match placement::plan_with(&q2k_attn, &machine, 4096, &NoKv, None) {
        Err(e @ PlacementError::NoCardFormat { .. }) => {
            let PlacementError::NoCardFormat { name, ty, role } = &e else {
                unreachable!()
            };
            assert_eq!(
                (name.as_str(), *ty, *role),
                ("blk.0.attn_q_b.weight", GgmlType::Q2_K, Role::Attention)
            );
            let text = e.to_string();
            assert!(
                text.contains("blk.0.attn_q_b.weight") && text.contains("q2_K"),
                "{text}"
            );
        }
        Err(e) => panic!("refused with the wrong error: {e}"),
        Ok(_) => panic!("a q2_K attention tensor was placed on a card"),
    }

    for ty in [GgmlType::Q2_K, GgmlType::Q5_K] {
        let stack = synthetic_model(vec![synthetic(
            "blk.0.ffn_down_exps.weight",
            Some(0),
            Role::RoutedExperts,
            ty,
            &[4, 8],
        )]);
        let plan = placement::plan_with(&stack, &machine, 4096, &NoKv, None)
            .unwrap_or_else(|e| panic!("a {ty} routed stack plans, on the host: {e}"));
        assert!(
            plan.violations().is_empty(),
            "{ty}: {:?}",
            plan.violations()
        );
        assert_eq!(plan.n_l, vec![0], "{ty}");
        let row = &plan.rows[1];
        assert!(
            matches!(row.segments.as_slice(), [s] if s.device == Device::Host && s.format == Format::HostFile),
            "{ty}: {:?}",
            row.segments
        );
    }
}

/// Which routed stacks go on the card is the program's: under glm5next's
/// card experts ([`card_routed`]) a q5_K stack keeps experts on the card, in
/// its file bytes, where V4.1's rule ([`placement::plan_with`]) keeps it on
/// the host; a q6_K stack stays on the host under both.
#[test]
fn routed_card_format_is_the_programs() {
    let machine = workstation::plan_a(1);
    let levers = PlanLevers::default();
    for (ty, on_card) in [
        (GgmlType::Q4_K, true),
        (GgmlType::Q5_K, true),
        (GgmlType::Q6_K, false),
    ] {
        let stack = synthetic_model(vec![synthetic(
            "blk.0.ffn_down_exps.weight",
            Some(0),
            Role::RoutedExperts,
            ty,
            &[4, 8],
        )]);
        let plan = placement::plan_routed(&stack, &machine, 4096, &NoKv, &levers, card_routed)
            .unwrap_or_else(|e| panic!("a {ty} routed stack plans: {e}"));
        assert!(
            plan.violations().is_empty(),
            "{ty}: {:?}",
            plan.violations()
        );
        let row = &plan.rows[1];
        if on_card {
            assert_eq!(plan.n_l, vec![8], "{ty}");
            assert!(
                matches!(row.segments.as_slice(), [s] if s.device == Device::Card(0) && s.format == Format::Card(CardFormat::KQuant)),
                "{ty}: {:?}",
                row.segments
            );
        } else {
            assert_eq!(plan.n_l, vec![0], "{ty}");
            assert!(
                matches!(row.segments.as_slice(), [s] if s.device == Device::Host && s.format == Format::HostFile),
                "{ty}: {:?}",
                row.segments
            );
        }
    }
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
    let plan = placement::plan_with(&model, &machine, 4096, &NoKv, None)
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

/// `layers` layers of one q4_K routed stack each — 8 experts of 4 rows of
/// 256 values, 576 B an expert — between the token embedding and a q8_0
/// head (4,352 B on its card).
fn layered_model(layers: usize) -> ModelTensors {
    let mut tensors = vec![synthetic(
        "token_embd.weight",
        None,
        Role::TokenEmbedding,
        GgmlType::Q8_0,
        &[16],
    )];
    for l in 0..layers {
        tensors.push(synthetic(
            &format!("blk.{l}.ffn_up_exps.weight"),
            Some(l),
            Role::RoutedExperts,
            GgmlType::Q4_K,
            &[4, 8],
        ));
    }
    tensors.push(synthetic(
        "output.weight",
        None,
        Role::Head,
        GgmlType::Q8_0,
        &[16],
    ));
    ModelTensors {
        tensors,
        layers,
        experts: 8,
        experts_used: 2,
    }
}

/// A card of `usable` bytes on granule 1, so every buffer costs its bytes:
/// running `layers` with the head, or a tier when `layers` is empty.
fn byte_card(name: &str, usable: u64, layers: Range<usize>) -> Card {
    Card {
        name: name.to_string(),
        device: None,
        usable_bytes: usable,
        context_bytes: 0,
        scratch_bytes: 0,
        margin_bytes: 0,
        granule_bytes: NonZeroU64::MIN,
        head: !layers.is_empty(),
        layers,
        token_embedding: false,
        reserves: Vec::new(),
    }
}

const HEAD_Q8: u64 = 4_352;
const EXPERT_Q4K: u64 = 576;

/// Each card's experts per layer, by device, from the rows.
fn held_ids(plan: &Plan<'_>, layer: usize) -> Vec<(Device, Vec<u32>)> {
    plan.rows
        .iter()
        .filter(|r| plan.model.tensors[r.tensor].layer == Some(layer))
        .flat_map(|r| {
            r.segments.iter().map(|s| {
                (
                    s.device,
                    s.experts.as_ref().map_or(Vec::new(), |e| e.ids().to_vec()),
                )
            })
        })
        .collect()
}

/// The tier rule on three layers: the stage card keeps 5 experts in the id
/// order's cycle (n_l 2, 2, 1) exactly as with no tier; the first tier fits
/// 4 and fills the layer with the fewest first (1, 1, 2 — every total 3);
/// the second tier, with a reserve of one expert's bytes, fits 3 after both
/// (1, 1, 1); each takes the ids after the ones before it; the host holds
/// the rest; the tiers' reserves count in their totals; no invariant breaks.
#[test]
fn tier_takes_the_next_ranks_evenly() {
    let model = layered_model(3);
    let stage = byte_card("stage", HEAD_Q8 + 5 * EXPERT_Q4K, 0..3);
    let mut t2 = byte_card("tier2", 4 * EXPERT_Q4K, 0..0);
    t2.reserves.push(("draft".to_string(), EXPERT_Q4K));
    let machine = Machine {
        cards: vec![stage.clone()],
        tiers: vec![byte_card("tier1", 4 * EXPERT_Q4K, 0..0), t2],
        host: workstation::host(),
    };
    let alone = Machine {
        cards: vec![stage],
        tiers: Vec::new(),
        host: workstation::host(),
    };
    let base = placement::plan_with(&model, &alone, 4096, &NoKv, None).expect("no tier");
    let plan = placement::plan_with(&model, &machine, 4096, &NoKv, None).expect("tiers");
    assert!(plan.violations().is_empty(), "{:?}", plan.violations());
    assert_eq!(base.n_l, vec![2, 2, 1]);
    assert_eq!(plan.n_l, base.n_l, "the stage card plans as with no tier");
    assert_eq!(card_line(&plan.cards[0]), card_line(&base.cards[0]));
    assert_eq!(plan.tier_n_l, vec![vec![1, 1, 2], vec![1, 1, 1]]);
    let experts: Vec<u64> = plan.cards.iter().map(|c| c.experts).collect();
    assert_eq!(experts, vec![5, 4, 3]);
    assert_eq!(plan.cards[2].reserve_bytes, EXPERT_Q4K);
    assert_eq!(plan.cards[2].headroom_bytes, 0);
    assert_eq!(plan.host.experts, 3 * 8 - 12);
    assert_eq!(
        held_ids(&plan, 2),
        vec![
            (Device::Card(0), vec![0]),
            (Device::Card(1), vec![1, 2]),
            (Device::Card(2), vec![3]),
            (Device::Host, vec![4, 5, 6, 7]),
        ]
    );
}

/// A tier card runs no stage: one with layers, the head or the token
/// embedding is refused by name, before anything is planned.
#[test]
fn tier_card_runs_no_stage() {
    let model = layered_model(1);
    let stage = byte_card("stage", HEAD_Q8, 0..1);
    for (tier, what) in [
        (byte_card("t", 0, 0..1), "it runs layers 0..1"),
        (
            Card {
                head: true,
                ..byte_card("t", 0, 0..0)
            },
            "it carries the head",
        ),
        (
            Card {
                token_embedding: true,
                ..byte_card("t", 0, 0..0)
            },
            "it holds the token embedding",
        ),
    ] {
        let machine = Machine {
            cards: vec![stage.clone()],
            tiers: vec![tier],
            host: workstation::host(),
        };
        match placement::plan_with(&model, &machine, 4096, &NoKv, None) {
            Err(e @ PlacementError::Tier { .. }) => {
                assert_eq!(
                    e.to_string(),
                    format!("tier card t: {what}; a tier card runs no stage")
                );
            }
            Err(e) => panic!("{what}: {e}, not the tier refusal"),
            Ok(_) => panic!("{what}: the tier was planned"),
        }
    }
}

/// An expert on two cards is refused by name, with the tensor, the expert and
/// both cards; disjoint lists split the stack with the rest on the host.
#[test]
fn expert_on_two_cards_is_refused_by_name() {
    let model = layered_model(1);
    let t = &model.tensors[1];
    let list = |ids: Vec<u32>| ExpertList::new(ids, 8).expect("a list");
    match placement::routed_row_on(
        1,
        t,
        0,
        vec![(0, list(vec![0, 1])), (1, list(vec![1, 2]))],
        &model,
    ) {
        Err(e @ PlacementError::ExpertOnTwoCards { .. }) => assert_eq!(
            e.to_string(),
            "tensor blk.0.ffn_up_exps.weight: expert 1 is on card #0 and on card #1"
        ),
        Err(e) => panic!("{e}, not the two-card refusal"),
        Ok(r) => panic!("an expert on two cards was placed: {:?}", r.segments),
    }
    let row = placement::routed_row_on(
        1,
        t,
        0,
        vec![(0, list(vec![0, 1])), (1, list(vec![2]))],
        &model,
    )
    .expect("disjoint lists");
    let got: Vec<(Device, String)> = row
        .segments
        .iter()
        .map(|s| {
            (
                s.device,
                s.experts
                    .as_ref()
                    .map_or(String::new(), ToString::to_string),
            )
        })
        .collect();
    assert_eq!(
        got,
        vec![
            (Device::Card(0), "0..2".to_string()),
            (Device::Card(1), "2..3".to_string()),
            (Device::Host, "3..8".to_string()),
        ]
    );
}
