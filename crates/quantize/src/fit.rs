//! `--fit`: a rules file for a byte budget, chosen from measured sensitivity.
//!
//! A mix is a type for each *unit*: a tensor class ([`CLASSES`]: the stems that share one choice, like the routed
//! gate and up stacks) over a layer range. [`ladder`] lists every unit of a model with the types it may take (the
//! reader's list for the stem, the types the quantizer writes, the blocks the rows divide into) and their bytes;
//! `--fit-ladder` prints it, one rule token per row. A sensitivity table ([`Table`]) holds, per unit and type, the
//! mean KLD of the model with that unit alone at that type against the base mix (`tools/ref/qfit.py` measures it
//! through the engine's own loss tool). [`select`] starts every unit at its largest candidate and takes the steps
//! with the least KLD per byte saved until the file fits; [`run`] ties the pieces to a plan and writes the rules.
//!
//! The selection is the multiple-choice knapsack's greedy over each unit's lower convex hull of (bytes, KLD): the
//! steps' order is fixed by their slopes, so a larger budget stops earlier in the same order and every unit's pick is
//! monotone in the budget. The predicted loss is the sum of the units' table values (the first-order additive model);
//! `tools/ref/qfit.py verify` measures the picked mix against it.

use crate::cli::{Args, Fit};
use crate::plan::{Item, Plan};
use crate::resolve::{layer_of, parse_type};
use crate::rules::Rules;
use crate::run::Prepared;
use crate::scan::row_bytes;
use crate::{Error, Result};
use gguf::quant::GgmlType;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

/// A tensor class: the stems (a tensor's name past `blk.N.`, the whole name for a model-level tensor) that take one
/// type together.
#[derive(Debug)]
pub struct Class {
    pub name: &'static str,
    pub stems: &'static [&'static str],
}

/// Every class a unit can be of. The routed gate and up are one class: the card's fused gate-up entry reads both in
/// one type.
pub const CLASSES: &[Class] = &[
    Class {
        name: "routed_gate_up",
        stems: &["ffn_gate_exps.weight", "ffn_up_exps.weight"],
    },
    Class {
        name: "routed_down",
        stems: &["ffn_down_exps.weight"],
    },
    Class {
        name: "attn",
        stems: &[
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
        ],
    },
    Class {
        name: "shared",
        stems: &[
            "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
        ],
    },
    Class {
        name: "dense_ffn",
        stems: &["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"],
    },
    Class {
        name: "nextn",
        stems: &["nextn.eh_proj.weight"],
    },
    Class {
        name: "head",
        stems: &["output.weight"],
    },
    Class {
        name: "embed",
        stems: &["token_embd.weight"],
    },
];

/// The class named `name`.
#[must_use]
pub fn class_named(name: &str) -> Option<&'static Class> {
    CLASSES.iter().find(|c| c.name == name)
}

/// A tensor's stem: its name past `blk.N.`, the whole name when it has no block prefix.
#[must_use]
pub fn stem_of(name: &str) -> &str {
    match layer_of(name) {
        Some(_) => name
            .strip_prefix("blk.")
            .and_then(|r| r.split_once('.'))
            .map_or(name, |(_, s)| s),
        None => name,
    }
}

/// The class a tensor is of.
#[must_use]
pub fn class_of(name: &str) -> Option<&'static Class> {
    let stem = stem_of(name);
    CLASSES.iter().find(|c| c.stems.contains(&stem))
}

/// The types a fit can name: the ones the quantizer writes with a block rule.
pub const FIT_TYPES: [GgmlType; 6] = [
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
    GgmlType::Q8_0,
    GgmlType::IQ2_XXS,
    GgmlType::IQ1_S,
];

/// The types the engine's reader runs the stem in for architecture `arch`, `None` where the architecture has no table
/// or the table has no such stem.
fn reader_types(arch: &str, stem: &str) -> Option<&'static [GgmlType]> {
    if gguf::arch::kimi_k3::is_arch(arch) {
        gguf::arch::kimi_k3::allowed_types(stem)
    } else {
        gguf::arch::allowed_types(arch, stem)
    }
}

/// A unit: a class (by its index in [`CLASSES`]) over the layers `first..=last`, or a model-level class with no layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnitKey {
    class: usize,
    pub layers: Option<(i64, i64)>,
}

impl UnitKey {
    /// A unit of `class` over `layers`.
    #[must_use]
    pub fn new(class: &'static Class, layers: Option<(i64, i64)>) -> UnitKey {
        let class = CLASSES
            .iter()
            .position(|c| c.name == class.name)
            .expect("a class is a row of CLASSES");
        UnitKey { class, layers }
    }

    /// The unit's class.
    #[must_use]
    pub fn class(&self) -> &'static Class {
        &CLASSES[self.class]
    }

    /// The layers column: `first-last`, or `-` for a model-level unit.
    #[must_use]
    pub fn layers_text(&self) -> String {
        match self.layers {
            Some((a, b)) => format!("{a}-{b}"),
            None => "-".to_string(),
        }
    }

    /// Whether the tensor `name` is in the unit.
    #[must_use]
    pub fn holds(&self, name: &str) -> bool {
        class_of(name).is_some_and(|c| c.name == self.class().name)
            && match (self.layers, layer_of(name)) {
                (Some((a, b)), Some(l)) => (a..=b).contains(&l),
                (None, None) => true,
                _ => false,
            }
    }

    /// The rules-file token that types the unit's tensors `ty`: a regex over the lowercased name the rule reader
    /// takes, `=` and the type.
    #[must_use]
    pub fn rule(&self, ty: GgmlType) -> String {
        let stems: Vec<String> = self
            .class()
            .stems
            .iter()
            .map(|s| s.replace('.', "\\."))
            .collect();
        let stems = stems.join("|");
        let name = type_name(ty).to_ascii_lowercase();
        match self.layers {
            Some((a, b)) => {
                let layers: Vec<String> = (a..=b).map(|l| l.to_string()).collect();
                format!("^blk\\.({})\\.({stems})$={name}", layers.join("|"))
            }
            None => format!("^({stems})$={name}"),
        }
    }
}

impl std::fmt::Display for UnitKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.layers {
            Some(_) => write!(f, "{}@{}", self.class().name, self.layers_text()),
            None => write!(f, "{}", self.class().name),
        }
    }
}

/// A type's name as the tables and the reports spell it: `Q4_K`, `IQ1_S`.
#[must_use]
pub fn type_name(ty: GgmlType) -> String {
    ty.to_string().to_ascii_uppercase()
}

/// The bytes a tensor stores under `ty`, padded to the file's alignment of 32.
fn stored(it: &Item, ty: GgmlType) -> Option<u64> {
    let ne0 = usize::try_from(*it.dims.first()?).ok()?;
    let rows: u64 = it.dims[1..].iter().product();
    Some(pad(row_bytes(ty, ne0)? as u64 * rows))
}

/// `bytes` padded to the alignment the writer keeps ([`crate::plan::Plan::build`] refuses any other).
fn pad(bytes: u64) -> u64 {
    bytes.next_multiple_of(32)
}

/// The fit types the reader and the rows allow for every tensor of `tensors`, in [`FIT_TYPES`]' order.
fn legal(arch: &str, tensors: &[&Item]) -> Result<Vec<GgmlType>> {
    let mut out: Vec<GgmlType> = FIT_TYPES.to_vec();
    for it in tensors {
        let stem = stem_of(&it.name);
        let allowed = reader_types(arch, stem).ok_or_else(|| {
            Error::refused(format!(
                "{}: the {arch} reader has no type list for {stem}, so no unit can hold it",
                it.name
            ))
        })?;
        out.retain(|t| allowed.contains(t) && stored(it, *t).is_some());
    }
    Ok(out)
}

/// One candidate type of a unit.
#[derive(Debug, Clone, Copy)]
pub struct Cand {
    pub ty: GgmlType,
    /// The unit's tensors' stored bytes at this type.
    pub bytes: u64,
    /// The table's mean KLD of the model with this unit alone at the type (0 in a ladder).
    pub dkld: f64,
    pub se: f64,
}

/// A unit with the plan's tensors it holds and its candidates.
#[derive(Debug, Clone)]
pub struct Unit {
    pub key: UnitKey,
    /// Indices into [`Plan::items`].
    pub tensors: Vec<usize>,
    pub cands: Vec<Cand>,
}

/// Every unit of the model with the types it may take, and the classes with one legal type only (no choice to make).
#[derive(Debug)]
pub struct Ladder {
    pub units: Vec<Unit>,
    pub single: Vec<&'static str>,
}

/// The units of `plan` for architecture `arch`, `group` layers to a unit (0: one unit over all of a class's layers).
pub fn ladder(plan: &Plan, arch: &str, group: usize) -> Result<Ladder> {
    // class -> group index -> tensors
    let mut by: BTreeMap<(usize, i64), Vec<usize>> = BTreeMap::new();
    for (i, it) in plan.items.iter().enumerate() {
        let Some(class) = class_of(&it.name) else {
            continue;
        };
        let at = UnitKey::new(class, None).class;
        let g = match (layer_of(&it.name), group) {
            (Some(l), g) if g > 0 => l / g as i64,
            _ => 0,
        };
        by.entry((at, g)).or_default().push(i);
    }
    let mut units = Vec::new();
    let mut single = Vec::new();
    for ((at, _), tensors) in by {
        let class = &CLASSES[at];
        let items: Vec<&Item> = tensors.iter().map(|&i| &plan.items[i]).collect();
        let layers = items.iter().filter_map(|it| layer_of(&it.name)).fold(
            None,
            |r: Option<(i64, i64)>, l| match r {
                None => Some((l, l)),
                Some((a, b)) => Some((a.min(l), b.max(l))),
            },
        );
        let types = legal(arch, &items)?;
        if types.len() < 2 {
            if !single.contains(&class.name) {
                single.push(class.name);
            }
            continue;
        }
        let mut cands = Vec::new();
        for ty in types {
            let bytes = items
                .iter()
                .map(|it| stored(it, ty).expect("legal types are storable"))
                .sum();
            cands.push(Cand {
                ty,
                bytes,
                dkld: 0.0,
                se: 0.0,
            });
        }
        cands.sort_by_key(|c| std::cmp::Reverse(c.bytes));
        units.push(Unit {
            key: UnitKey::new(class, layers),
            tensors,
            cands,
        });
    }
    units.sort_by_key(|u| u.key);
    Ok(Ladder { units, single })
}

/// What `--fit-ladder` prints: `plan_bytes` is the file the plan under the command's own rules writes.
#[must_use]
pub fn render_ladder(l: &Ladder, group: usize, plan_bytes: u64) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "# bloomery-quantize --fit-ladder: {} units, {} layer(s) a unit (0: a class over all its layers), base plan {plan_bytes} B; a class with one legal type is no unit: {}",
        l.units.len(),
        group,
        if l.single.is_empty() {
            "none".to_string()
        } else {
            l.single.join(", ")
        }
    );
    let _ = writeln!(s, "class\tlayers\ttype\tbytes\trule");
    for u in &l.units {
        for c in &u.cands {
            let _ = writeln!(
                s,
                "{}\t{}\t{}\t{}\t{}",
                u.key.class().name,
                u.key.layers_text(),
                type_name(c.ty),
                c.bytes,
                u.key.rule(c.ty)
            );
        }
    }
    s
}

/// One row of a sensitivity table.
#[derive(Debug, Clone)]
pub struct Row {
    pub key: UnitKey,
    pub ty: GgmlType,
    pub dkld: f64,
    pub se: f64,
    pub positions: u64,
}

/// A sensitivity table: `class layers type dkld se positions`, tab separated, `#` lines free text.
#[derive(Debug)]
pub struct Table {
    pub rows: Vec<Row>,
}

const TABLE_HEAD: &str = "class\tlayers\ttype\tdkld\tse\tpositions";

impl Table {
    /// Read a table's text (`what` names it in a refusal).
    pub fn parse(text: &str, what: &str) -> Result<Table> {
        let bad = |n: usize, why: String| Error::refused(format!("{what} line {n}: {why}"));
        let mut rows: Vec<Row> = Vec::new();
        let mut head = false;
        for (i, line) in text.lines().enumerate() {
            let n = i + 1;
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            if !head {
                if line != TABLE_HEAD {
                    return Err(bad(
                        n,
                        format!("the first row is {line:?}, the header is {TABLE_HEAD:?}"),
                    ));
                }
                head = true;
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            let [class, layers, ty, dkld, se, positions] = f[..] else {
                return Err(bad(n, format!("{} columns, the header has 6", f.len())));
            };
            let class = class_named(class).ok_or_else(|| {
                let names: Vec<&str> = CLASSES.iter().map(|c| c.name).collect();
                bad(
                    n,
                    format!("class {class:?} is none of {}", names.join(", ")),
                )
            })?;
            let layers = match layers {
                "-" => None,
                s => {
                    let (a, b) = s.split_once('-').unwrap_or((s, s));
                    match (a.parse::<i64>(), b.parse::<i64>()) {
                        (Ok(a), Ok(b)) if 0 <= a && a <= b => Some((a, b)),
                        _ => {
                            return Err(bad(
                                n,
                                format!("layers {s:?} is not `-`, `a` or `a-b` with 0 <= a <= b"),
                            ));
                        }
                    }
                }
            };
            let ty = parse_type(ty).map_err(|e| bad(n, e.to_string()))?;
            if !FIT_TYPES.contains(&ty) {
                return Err(bad(
                    n,
                    format!("type {} is not one a fit can choose", type_name(ty)),
                ));
            }
            let num = |s: &str, what: &str| -> Result<f64> {
                s.parse::<f64>()
                    .ok()
                    .filter(|v| v.is_finite() && *v >= 0.0)
                    .ok_or_else(|| {
                        bad(
                            n,
                            format!("{what} {s:?} is not a finite number of at least 0"),
                        )
                    })
            };
            let dkld = num(dkld, "dkld")?;
            let se = num(se, "se")?;
            let positions = positions
                .parse::<u64>()
                .ok()
                .filter(|p| *p > 0)
                .ok_or_else(|| bad(n, format!("positions {positions:?} is not a count above 0")))?;
            let key = UnitKey::new(class, layers);
            if rows.iter().any(|r| r.key == key && r.ty == ty) {
                return Err(bad(n, format!("{key} at {} is given twice", type_name(ty))));
            }
            rows.push(Row {
                key,
                ty,
                dkld,
                se,
                positions,
            });
        }
        if !head {
            return Err(Error::refused(format!(
                "{what}: no header row ({TABLE_HEAD:?})"
            )));
        }
        if rows.is_empty() {
            return Err(Error::refused(format!("{what}: the table holds no row")));
        }
        Ok(Table { rows })
    }
}

/// The units a table names over a plan, the bytes of everything else and the file's own overhead.
#[derive(Debug)]
pub struct Fitted {
    pub units: Vec<Unit>,
    /// The stored bytes of every tensor no unit holds, and the headers: what the file holds at any pick.
    pub fixed: u64,
}

/// The table's units over `plan`: each holds at least one tensor, no tensor is in two, every tensor of a class the
/// table names is in one, and every row's type is one the reader and the rows allow.
pub fn units_for(plan: &Plan, arch: &str, table: &Table, a: &Args) -> Result<Fitted> {
    let mut by: BTreeMap<UnitKey, Vec<&Row>> = BTreeMap::new();
    for r in &table.rows {
        by.entry(r.key).or_default().push(r);
    }
    let mut claimed: Vec<Option<UnitKey>> = vec![None; plan.items.len()];
    let mut units = Vec::new();
    for (key, rows) in &by {
        let tensors: Vec<usize> = (0..plan.items.len())
            .filter(|&i| key.holds(&plan.items[i].name))
            .collect();
        if tensors.is_empty() {
            return Err(Error::refused(format!(
                "the table's unit {key} holds no tensor of this model"
            )));
        }
        for &i in &tensors {
            let name = &plan.items[i].name;
            if let Some(other) = claimed[i] {
                return Err(Error::refused(format!(
                    "{name} is in two units of the table: {other} and {key}"
                )));
            }
            claimed[i] = Some(*key);
            let flag = match name.as_str() {
                "output.weight" => a.output_tensor_type,
                "token_embd.weight" => a.token_embedding_type,
                _ => None,
            };
            if let Some(t) = flag {
                return Err(Error::refused(format!(
                    "{name} is typed {t} by its flag (--output-tensor-type or --token-embedding-type), so the table's unit {key} cannot take effect"
                )));
            }
        }
        let items: Vec<&Item> = tensors.iter().map(|&i| &plan.items[i]).collect();
        let allowed = legal(arch, &items)?;
        let mut cands = Vec::new();
        for r in rows {
            if !allowed.contains(&r.ty) {
                let names: Vec<String> = allowed.iter().map(|t| type_name(*t)).collect();
                return Err(Error::refused(format!(
                    "the table's row {key} {}: the {arch} reader and the rows allow {} for it",
                    type_name(r.ty),
                    names.join(", ")
                )));
            }
            let bytes = items
                .iter()
                .map(|it| stored(it, r.ty).expect("legal types are storable"))
                .sum();
            cands.push(Cand {
                ty: r.ty,
                bytes,
                dkld: r.dkld,
                se: r.se,
            });
        }
        units.push(Unit {
            key: *key,
            tensors,
            cands,
        });
    }
    for (i, it) in plan.items.iter().enumerate() {
        let Some(class) = class_of(&it.name) else {
            continue;
        };
        if claimed[i].is_none() && units.iter().any(|u| u.key.class().name == class.name) {
            return Err(Error::refused(format!(
                "{} is of class {}, which the table covers for other layers only: no unit of it holds this tensor",
                it.name, class.name
            )));
        }
    }
    let all: u64 = plan.items.iter().map(|it| pad(it.bytes_out)).sum();
    let overhead = plan.out_file_bytes.checked_sub(all).ok_or_else(|| {
        Error::refused(format!(
            "the plan's file is {} B, under its tensors' {all} B",
            plan.out_file_bytes
        ))
    })?;
    let in_units: u64 = units
        .iter()
        .flat_map(|u| &u.tensors)
        .map(|&i| pad(plan.items[i].bytes_out))
        .sum();
    Ok(Fitted {
        units,
        fixed: all - in_units + overhead,
    })
}

/// What [`select`] chose.
#[derive(Debug, PartialEq)]
pub struct Pick {
    /// Per unit, the index of its chosen candidate.
    pub choice: Vec<usize>,
    /// The file's bytes at the pick: `fixed` and the units' chosen bytes.
    pub bytes: u64,
    /// The sum of the chosen candidates' table KLD.
    pub dkld: f64,
    /// Steps taken, of the steps there are.
    pub steps: usize,
    pub steps_total: usize,
}

/// The candidates on a unit's lower convex hull of (bytes, KLD), the largest first: a candidate another one at no more
/// bytes and no more KLD beats is dropped, then every point above the chord of its neighbours.
fn hull(cands: &[Cand]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..cands.len()).collect();
    order.sort_by(|&a, &b| {
        cands[a]
            .bytes
            .cmp(&cands[b].bytes)
            .then(cands[a].dkld.total_cmp(&cands[b].dkld))
            .then(a.cmp(&b))
    });
    // ascending bytes: keep a candidate only when it beats every smaller one
    let mut best = f64::INFINITY;
    let mut kept: Vec<usize> = Vec::new();
    for i in order {
        if cands[i].dkld < best {
            best = cands[i].dkld;
            kept.push(i);
        }
    }
    kept.reverse();
    // descending bytes, KLD rising: slopes (KLD per byte saved) must not fall
    let slope = |a: usize, b: usize| {
        (cands[b].dkld - cands[a].dkld) / (cands[a].bytes - cands[b].bytes) as f64
    };
    let mut h: Vec<usize> = Vec::new();
    for i in kept {
        while h.len() >= 2 && slope(h[h.len() - 2], h[h.len() - 1]) > slope(h[h.len() - 1], i) {
            h.pop();
        }
        h.push(i);
    }
    h
}

/// The pick for `budget` bytes of file: every unit at its largest candidate, then the steps of least KLD per byte
/// saved until `fixed` and the units fit. A budget below the smallest choice is refused with both numbers.
pub fn select(units: &[Unit], fixed: u64, budget: u64) -> Result<Pick> {
    struct Step {
        unit: usize,
        edge: usize,
        saved: u64,
        cost: f64,
    }
    let hulls: Vec<Vec<usize>> = units.iter().map(|u| hull(&u.cands)).collect();
    let mut steps: Vec<Step> = Vec::new();
    for (ui, h) in hulls.iter().enumerate() {
        for (edge, w) in h.windows(2).enumerate() {
            let (a, b) = (&units[ui].cands[w[0]], &units[ui].cands[w[1]]);
            steps.push(Step {
                unit: ui,
                edge,
                saved: a.bytes - b.bytes,
                cost: b.dkld - a.dkld,
            });
        }
    }
    steps.sort_by(|x, y| {
        (x.cost / x.saved as f64)
            .total_cmp(&(y.cost / y.saved as f64))
            .then(x.unit.cmp(&y.unit))
            .then(x.edge.cmp(&y.edge))
    });
    let mut level = vec![0usize; units.len()];
    let top = |level: &[usize]| -> (u64, f64) {
        units
            .iter()
            .enumerate()
            .fold((fixed, 0.0), |(b, d), (ui, u)| {
                let c = &u.cands[hulls[ui][level[ui]]];
                (b + c.bytes, d + c.dkld)
            })
    };
    let (mut bytes, mut dkld) = top(&level);
    let mut taken = 0;
    for s in &steps {
        if bytes <= budget {
            break;
        }
        level[s.unit] = s.edge + 1;
        bytes -= s.saved;
        dkld += s.cost;
        taken += 1;
    }
    if bytes > budget {
        return Err(Error::refused(format!(
            "--fit {budget} B cannot be met: the smallest file this table can pick is {bytes} B ({fixed} B that no unit holds and every unit at its smallest type)"
        )));
    }
    Ok(Pick {
        choice: level
            .iter()
            .enumerate()
            .map(|(ui, &l)| hulls[ui][l])
            .collect(),
        bytes,
        dkld,
        steps: taken,
        steps_total: steps.len(),
    })
}

/// The rules file of a pick: one token a unit, then the base rules' text, one token a line. A rule file holds no
/// comment, so none is written.
#[must_use]
pub fn rules_text(units: &[Unit], pick: &Pick, base: Option<&str>) -> String {
    let mut s = String::new();
    for (u, &c) in units.iter().zip(&pick.choice) {
        let _ = writeln!(s, "{}", u.key.rule(u.cands[c].ty));
    }
    for tok in base.unwrap_or("").split_whitespace() {
        let _ = writeln!(s, "{tok}");
    }
    s
}

/// A byte count: whole bytes, or a number with a unit (`B`, decimal `KB MB GB TB`, binary `KiB MiB GiB TiB`).
pub fn parse_bytes(s: &str) -> Result<u64> {
    let bad = || {
        Error::Usage(format!(
            "--fit {s:?}: a byte count is a number with an optional unit (B, KB, MB, GB, TB, KiB, MiB, GiB, TiB); bloomery-quantize --help prints the usage"
        ))
    };
    let at = s
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(s.len());
    let (num, unit) = s.split_at(at);
    let mult: u64 = match unit.to_ascii_uppercase().as_str() {
        "" | "B" => 1,
        "KB" => 1_000,
        "MB" => 1_000_000,
        "GB" => 1_000_000_000,
        "TB" => 1_000_000_000_000,
        "KIB" => 1 << 10,
        "MIB" => 1 << 20,
        "GIB" => 1 << 30,
        "TIB" => 1 << 40,
        _ => return Err(bad()),
    };
    let bytes = match num.split_once('.') {
        None => num
            .parse::<u64>()
            .ok()
            .and_then(|n| n.checked_mul(mult))
            .ok_or_else(bad)?,
        Some(_) => {
            let v: f64 = num.parse().map_err(|_| bad())?;
            let b = (v * mult as f64).round();
            if !(b.is_finite() && b >= 1.0 && b < u64::MAX as f64) {
                return Err(bad());
            }
            b as u64
        }
    };
    if bytes == 0 {
        return Err(bad());
    }
    Ok(bytes)
}

fn io_err(p: &Path, source: std::io::Error) -> Error {
    Error::Io {
        path: p.display().to_string(),
        source,
    }
}

/// The report of a fit: the picks, the bytes and the loss the table predicts.
fn render_fit(fitted: &Fitted, pick: &Pick, budget: u64, rules_bytes: u64) -> String {
    let mut s = String::new();
    let gb = |b: u64| b as f64 / 1e9;
    let _ = writeln!(
        s,
        "bloomery-quantize --fit: budget {budget} B ({:.3} GB), {} units, {} B in no unit and in the headers",
        gb(budget),
        fitted.units.len(),
        fitted.fixed
    );
    let _ = writeln!(s, "  unit\ttype\tbytes\tdkld\tse");
    for (u, &c) in fitted.units.iter().zip(&pick.choice) {
        let c = &u.cands[c];
        let _ = writeln!(
            s,
            "  {}\t{}\t{}\t{:.6}\t{:.6}",
            u.key,
            type_name(c.ty),
            c.bytes,
            c.dkld,
            c.se
        );
    }
    let _ = writeln!(
        s,
        "fit: {} of {} steps; the file is {} B ({:.3} GB), {} B under the budget; predicted sum of the units' dkld {:.6}",
        pick.steps,
        pick.steps_total,
        pick.bytes,
        gb(pick.bytes),
        budget - pick.bytes,
        pick.dkld
    );
    let _ = writeln!(
        s,
        "rules: {rules_bytes} B written; the plan under them holds the same {} B",
        pick.bytes
    );
    s
}

/// `--fit-ladder` and `--fit`: the text to print. A `--fit` run writes its rules file after the plan under it has
/// agreed with the pick to the byte, so a refusal leaves no file.
pub fn run(a: &Args, prep: &Prepared, fit: &Fit) -> Result<String> {
    let arch = prep
        .set
        .architecture()
        .ok_or_else(|| Error::refused("the input names no architecture"))?
        .to_string();
    match fit {
        Fit::Ladder { group } => Ok(render_ladder(
            &ladder(&prep.plan, &arch, *group)?,
            *group,
            prep.plan.out_file_bytes,
        )),
        Fit::Select { budget, table, out } => {
            let text = std::fs::read_to_string(table).map_err(|e| io_err(table, e))?;
            let table = Table::parse(&text, &table.display().to_string())?;
            let fitted = units_for(&prep.plan, &arch, &table, a)?;
            let pick = select(&fitted.units, fitted.fixed, *budget)?;
            let base = a
                .rules
                .as_ref()
                .map(|p| std::fs::read_to_string(p).map_err(|e| io_err(p, e)))
                .transpose()?;
            let rules = rules_text(&fitted.units, &pick, base.as_deref());
            let again = Plan::build(
                &*prep.set,
                &Rules::parse(&rules)?,
                None,
                prep.im.as_ref(),
                a,
            )?;
            if again.out_file_bytes != pick.bytes {
                return Err(Error::refused(format!(
                    "the fit predicted a file of {} B, the plan under its rules says {} B",
                    pick.bytes, again.out_file_bytes
                )));
            }
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(out)
                .map_err(|e| io_err(out, e))?;
            std::io::Write::write_all(&mut f, rules.as_bytes()).map_err(|e| io_err(out, e))?;
            Ok(render_fit(&fitted, &pick, *budget, rules.len() as u64))
        }
    }
}
