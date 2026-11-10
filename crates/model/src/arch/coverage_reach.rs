//! What a fixture cannot reach of a real file: the dispatch classes a header decides
//! ([`classes`]), the classes one description holds that another does not ([`diff`]), and, per
//! fixture, each class of a real file its fixture never runs and the reason ([`Reach::gaps`]),
//! held to the checked table [`TABLE`] ([`parse_table`], [`check_rows`]).
//!
//! A class is one of four kinds ([`ClassKind`]): a [`Need`] of the description (what a kernel
//! instance or a program step is built for), a type pin (a tensor stem and the type it is
//! stored in: the card entry the type's row names), a width (a matrix stem and its two leading
//! dims: the row width and rows a gemv or GEMM instance is chosen at), and a router row (the
//! shape table's instance for a mixture's rule, experts and pick count). A fixture is planned
//! from its source's header alone ([`fixture_header`]), so the whole comparison reads headers and
//! no tensor's bytes.
//!
//! What the classes cannot see is a host branch on run sizes (the prompt batch's GEMM threshold,
//! ctx segments, the slots a batch groups, the placement regime, the card count): the fixture's
//! arms and the real tier cover those.

use std::collections::BTreeSet;
use std::fmt;
use std::path::PathBuf;

use gguf::Split;
use models::shape::{MoeShape, select_router};
use models::{Ffn, needs};

use crate::arch::Read;
use crate::fixture::{FixtureSpec, Options, VariantRow};

/// The table of what each fixture cannot reach, relative to the repository root.
pub const TABLE: &str = "tools/kcover/unreached.tsv";

/// The table's columns, each once, in this order.
pub const COLUMNS: [&str; 6] = ["fixture", "real", "side", "kind", "class", "reason"];

/// The kind of a dispatch class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ClassKind {
    /// A [`Need`] of the description, by its text.
    Need,
    /// A tensor stem (the name past `blk.N.`, without `.weight`) and its stored type.
    Pin,
    /// A matrix stem and its first two dims (`dims[0]` the row width).
    Width,
    /// The router instance `models::shape::select_router` picks for a mixture, or its refusal.
    Router,
}

impl ClassKind {
    /// Every kind, in table order.
    pub const ALL: [ClassKind; 4] = [
        ClassKind::Need,
        ClassKind::Pin,
        ClassKind::Width,
        ClassKind::Router,
    ];

    /// The kind's word in the table.
    #[must_use]
    pub const fn word(self) -> &'static str {
        match self {
            ClassKind::Need => "need",
            ClassKind::Pin => "pin",
            ClassKind::Width => "width",
            ClassKind::Router => "router",
        }
    }

    /// The kind `word` names.
    #[must_use]
    pub fn of_word(word: &str) -> Option<ClassKind> {
        Self::ALL.into_iter().find(|k| k.word() == word)
    }
}

/// One dispatch class a header decides.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Class {
    pub kind: ClassKind,
    pub name: String,
}

impl fmt::Display for Class {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.kind.word(), self.name)
    }
}

/// The stem of a tensor: its name past `blk.N.` for a layer's, without a trailing `.weight`.
fn stem(name: &str, layer: Option<usize>) -> &str {
    let past = layer
        .and_then(|l| name.strip_prefix(&format!("blk.{l}.")))
        .unwrap_or(name);
    past.strip_suffix(".weight").unwrap_or(past)
}

/// Every dispatch class `read` decides, each once.
#[must_use]
pub fn classes(read: &Read) -> BTreeSet<Class> {
    let mut out = BTreeSet::new();
    let mut add = |kind, name: String| {
        out.insert(Class { kind, name });
    };
    for (need, _) in needs(&read.spec) {
        add(ClassKind::Need, need.to_string());
    }
    for t in &read.tensors.tensors {
        let s = stem(&t.name, t.layer);
        let ty =
            t.ty.name()
                .map_or_else(|| format!("type {}", t.ty.as_u32()), str::to_string);
        add(ClassKind::Pin, format!("{s} {ty}"));
        if let [rows_of, rows, ..] = t.dims[..] {
            add(ClassKind::Width, format!("{s} {rows_of}x{rows}"));
        }
    }
    for layer in &read.spec.layers {
        if let Ffn::Moe(m) = &layer.ffn {
            let shape = MoeShape::of(m);
            let name = match select_router(shape) {
                Ok(r) => format!(
                    "{:?} over {} experts, top {}..={}",
                    r.body, r.experts, r.top_k_min, r.top_k_max
                ),
                Err(e) => format!("refused: {e}"),
            };
            add(ClassKind::Router, name);
        }
    }
    out
}

/// Which description holds a class the other does not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Side {
    /// The real file holds it and the fixture does not: the fixture tier never runs it.
    Real,
    /// The fixture holds it and the real file does not: the fixture tier runs an entry the real
    /// file never does.
    Fixture,
}

impl Side {
    /// The side's word in the table.
    #[must_use]
    pub const fn word(self) -> &'static str {
        match self {
            Side::Real => "real",
            Side::Fixture => "fixture",
        }
    }

    /// The side `word` names.
    #[must_use]
    pub fn of_word(word: &str) -> Option<Side> {
        [Side::Real, Side::Fixture]
            .into_iter()
            .find(|s| s.word() == word)
    }
}

/// The classes one description holds that the other does not.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Diff {
    /// In the real file, not in the fixture.
    pub real: BTreeSet<Class>,
    /// In the fixture, not in the real file.
    pub fixture: BTreeSet<Class>,
}

impl Diff {
    /// Whether both sides are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.real.is_empty() && self.fixture.is_empty()
    }
}

/// Why a fixture cannot be held against a real file, or a row of [`TABLE`] is refused.
#[derive(Debug, thiserror::Error)]
pub enum ReachError {
    #[error("the real file is {real:?} and the fixture {fixture:?}: a diff holds one architecture")]
    Architecture {
        real: models::Arch,
        fixture: models::Arch,
    },
    #[error("the fixture's plan: {0}")]
    Plan(#[from] crate::fixture::FixtureError),
    #[error("the fixture's header does not lay out: {0}")]
    Layout(#[from] gguf::LoadError),
    #[error("reading the {what}: {detail}")]
    Read { what: &'static str, detail: String },
    #[error("{TABLE} line {line}: {detail}")]
    Table { line: usize, detail: String },
}

/// The classes `real` holds that `fixture` does not, and the other way.
///
/// # Errors
/// [`ReachError::Architecture`] for two descriptions of different architectures.
pub fn diff(real: &Read, fixture: &Read) -> Result<Diff, ReachError> {
    if real.spec.arch != fixture.spec.arch {
        return Err(ReachError::Architecture {
            real: real.spec.arch,
            fixture: fixture.spec.arch,
        });
    }
    let (r, f) = (classes(real), classes(fixture));
    Ok(Diff {
        real: r.difference(&f).cloned().collect(),
        fixture: f.difference(&r).cloned().collect(),
    })
}

/// The header of the fixture `spec` plans from `source`, as one shard holding no tensor's bytes:
/// every tensor in one shard (the shard cut decides no dispatch) and a card budget recorded as
/// 0, so a planned family's budget search does not run (the budget decides no class either).
///
/// # Errors
/// The plan's refusal, or a layout the reader refuses.
pub fn fixture_header(spec: &FixtureSpec, source: &Split) -> Result<Split, ReachError> {
    let opts = Options {
        shard_bytes: u64::MAX,
        card_budget: Some(0),
        ..spec.options()
    };
    let plan = crate::fixture::plan(spec, source, None, &opts)?;
    let mut layouts = plan.target.layouts()?;
    let (name, layout) = layouts.swap_remove(0);
    Ok(Split::header_only(PathBuf::from(name), &layout)?)
}

/// `split`'s description, a reader's refusal named by `what`.
fn read_of(split: &Split, what: &'static str) -> Result<Read, ReachError> {
    crate::arch::spec(split).map_err(|e| ReachError::Read {
        what,
        detail: e.to_string(),
    })
}

/// Why a fixture does not reach a class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Reason {
    /// The fixture's own source agrees with the real file on the class; the cut (its layer map,
    /// its mini widths and counts) does not.
    Cut,
    /// The cut's narrowed routed ff alone: a plan of the same map with the source's ff agrees with
    /// the real file on the class.
    Ff,
    /// The fixture's source disagrees with the real file on the class: the class is a fact of
    /// another file of the family (its quant mix, its shapes), not of the cut.
    Source,
}

impl Reason {
    /// The reason's word in the table.
    #[must_use]
    pub const fn word(self) -> &'static str {
        match self {
            Reason::Cut => "cut",
            Reason::Ff => "ff",
            Reason::Source => "source",
        }
    }

    /// The reason `word` names.
    #[must_use]
    pub fn of_word(word: &str) -> Option<Reason> {
        [Reason::Cut, Reason::Ff, Reason::Source]
            .into_iter()
            .find(|r| r.word() == word)
    }
}

/// One class a fixture does not reach of one real file, or runs that the file never does.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Gap {
    pub side: Side,
    pub class: Class,
    pub reason: Reason,
}

/// A fixture's classes beside its source's, the one comparison [`Reach::gaps`] makes per real
/// file.
pub struct Reach {
    /// The fixture's directory: the table's `fixture` column.
    pub fixture: String,
    /// The architecture the fixture declares: the variant rows of the same `family` reach it.
    pub arch: &'static str,
    /// The fixture's own description.
    pub read: Read,
    source: BTreeSet<Class>,
    /// The classes of the same map planned at the source's ff; `None` when the spec keeps it.
    wide: Option<BTreeSet<Class>>,
}

impl Reach {
    /// The fixture `spec` plans from `source`, under its directory `fixture`.
    ///
    /// # Errors
    /// The plan's or a reader's refusal, of the fixture, of the wide-ff plan or of the source.
    pub fn of(fixture: &str, spec: &FixtureSpec, source: &Split) -> Result<Reach, ReachError> {
        let read = read_of(&fixture_header(spec, source)?, "fixture")?;
        let source_read = read_of(source, "fixture's source")?;
        let wide = match spec.ff {
            None => None,
            Some(_) => {
                let wide = FixtureSpec {
                    ff: None,
                    ..spec.clone()
                };
                Some(classes(&read_of(
                    &fixture_header(&wide, source)?,
                    "fixture at the source's ff",
                )?))
            }
        };
        Ok(Reach {
            fixture: fixture.to_string(),
            arch: spec.arch,
            read,
            source: classes(&source_read),
            wide,
        })
    }

    /// Every class `real` holds that the fixture does not and the other way, each with its
    /// reason. A real-side type pin a variant row of the fixture's family retypes the fixture to
    /// (`variants`, `tools/fixture-variants.tsv`) is reached by that variant and is no gap.
    ///
    /// # Errors
    /// [`ReachError::Architecture`] for a real file of another architecture.
    pub fn gaps(&self, real: &Read, variants: &[VariantRow]) -> Result<Vec<Gap>, ReachError> {
        let d = diff(real, &self.read)?;
        let by_variant = |c: &Class| {
            c.kind == ClassKind::Pin
                && variants
                    .iter()
                    .filter(|v| v.family == self.arch)
                    .flat_map(|v| &v.map)
                    .any(|(role, ty)| ty.name().is_some_and(|t| c.name == format!("{role} {t}")))
        };
        let mut out = Vec::new();
        for c in d.real.into_iter().filter(|c| !by_variant(c)) {
            let reason = if !self.source.contains(&c) {
                Reason::Source
            } else if self.wide.as_ref().is_some_and(|w| w.contains(&c)) {
                Reason::Ff
            } else {
                Reason::Cut
            };
            out.push(Gap {
                side: Side::Real,
                class: c,
                reason,
            });
        }
        for c in d.fixture {
            let reason = if self.source.contains(&c) {
                Reason::Source
            } else if self.wide.as_ref().is_some_and(|w| !w.contains(&c)) {
                Reason::Ff
            } else {
                Reason::Cut
            };
            out.push(Gap {
                side: Side::Fixture,
                class: c,
                reason,
            });
        }
        out.sort();
        Ok(out)
    }
}

/// One row of [`TABLE`].
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Row {
    /// The fixture's directory (`tools/fixture-dirs.tsv`).
    pub fixture: String,
    /// The real header the row compares it with: `tests/<stem>` for the header tables checked in
    /// under `crates/model/tests/`, `census/<org>/<repo>/<set>` for a census stub set
    /// (`crates/refset/src/census.rs`).
    pub real: String,
    pub gap: Gap,
}

impl Row {
    /// The row as the table holds it, tab-separated, without the newline.
    #[must_use]
    pub fn line(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}",
            self.fixture,
            self.real,
            self.gap.side.word(),
            self.gap.class.kind.word(),
            self.gap.class.name,
            self.gap.reason.word()
        )
    }
}

/// The rows of the table `text`: `#` lines are comments, the first other line names
/// [`COLUMNS`], each once in that order, and every row has a value in each.
///
/// # Errors
/// [`ReachError::Table`] naming the line: a header that is not [`COLUMNS`], a row of another
/// width, an empty field, a side, kind or reason no word names, a row named twice.
pub fn parse_table(text: &str) -> Result<Vec<Row>, ReachError> {
    let mut lines = text
        .lines()
        .enumerate()
        .map(|(i, l)| (i + 1, l))
        .filter(|(_, l)| !l.starts_with('#'));
    let bad = |line, detail: String| ReachError::Table { line, detail };
    let Some((at, header)) = lines.next() else {
        return Err(bad(0, "holds no header line".into()));
    };
    let names: Vec<&str> = header.split('\t').collect();
    if names != COLUMNS {
        return Err(bad(
            at,
            format!("the header names {names:?}, the table's columns are {COLUMNS:?}"),
        ));
    }
    let mut rows: Vec<Row> = Vec::new();
    for (at, l) in lines {
        let f: Vec<&str> = l.split('\t').collect();
        let [fixture, real, side, kind, class, reason] = f[..] else {
            return Err(bad(
                at,
                format!("holds {} fields, the table {}", f.len(), COLUMNS.len()),
            ));
        };
        if let Some((col, _)) = COLUMNS.iter().zip(&f).find(|(_, v)| v.is_empty()) {
            return Err(bad(at, format!("{col} is empty")));
        }
        let side = Side::of_word(side).ok_or_else(|| bad(at, format!("side {side:?}")))?;
        let kind = ClassKind::of_word(kind).ok_or_else(|| bad(at, format!("kind {kind:?}")))?;
        let reason =
            Reason::of_word(reason).ok_or_else(|| bad(at, format!("reason {reason:?}")))?;
        let row = Row {
            fixture: fixture.to_string(),
            real: real.to_string(),
            gap: Gap {
                side,
                class: Class {
                    kind,
                    name: class.to_string(),
                },
                reason,
            },
        };
        if rows.iter().any(|r| {
            r.gap.side == side
                && r.gap.class == row.gap.class
                && r.fixture == row.fixture
                && r.real == row.real
        }) {
            return Err(bad(
                at,
                format!("names {} {} twice", side.word(), row.gap.class),
            ));
        }
        rows.push(row);
    }
    Ok(rows)
}

/// What differs between the table's rows of `fixture` against `real` and the gaps a run found:
/// one line per row the table holds that the run did not find and per gap the run found that the
/// table does not hold (a moved reason is both). Empty when they agree.
#[must_use]
pub fn check_rows(table: &[Row], fixture: &str, real: &str, gaps: &[Gap]) -> Vec<String> {
    let held: BTreeSet<&Gap> = table
        .iter()
        .filter(|r| r.fixture == fixture && r.real == real)
        .map(|r| &r.gap)
        .collect();
    let found: BTreeSet<&Gap> = gaps.iter().collect();
    let line = |g: &Gap| {
        Row {
            fixture: fixture.to_string(),
            real: real.to_string(),
            gap: g.clone(),
        }
        .line()
    };
    held.difference(&found)
        .map(|g| format!("the table holds a row the run does not find: {}", line(g)))
        .chain(
            found
                .difference(&held)
                .map(|g| format!("the run finds a row the table does not hold: {}", line(g))),
        )
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{COLUMNS, Class, ClassKind, Gap, Reason, Side, check_rows, parse_table};

    const HEADER: &str = "fixture\treal\tside\tkind\tclass\treason";

    fn gap(side: Side, name: &str, reason: Reason) -> Gap {
        Gap {
            side,
            class: Class {
                kind: ClassKind::Need,
                name: name.into(),
            },
            reason,
        }
    }

    /// The reader takes comments, the header and well-formed rows, and refuses by line a header
    /// of other columns, a row of another width, an empty field, a word no side, kind or reason
    /// names and a row named twice.
    #[test]
    fn the_table_reader_refuses_by_line() {
        assert_eq!(HEADER.split('\t').collect::<Vec<_>>(), COLUMNS);
        let row = "fx\ttests/a\treal\tneed\tdense ff 4\tcut";
        let rows = parse_table(&format!("# c\n{HEADER}\n{row}\n")).expect("a well-formed table");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].line(), row);
        for (what, text) in [
            ("no header", String::new()),
            ("other columns", format!("{}\tmore\n", HEADER)),
            (
                "a short row",
                format!("{HEADER}\nfx\ttests/a\treal\tneed\tcut\n"),
            ),
            (
                "an empty class",
                format!("{HEADER}\nfx\ttests/a\treal\tneed\t\tcut\n"),
            ),
            (
                "a side",
                format!("{HEADER}\nfx\ttests/a\tboth\tneed\tc\tcut\n"),
            ),
            (
                "a kind",
                format!("{HEADER}\nfx\ttests/a\treal\tshape\tc\tcut\n"),
            ),
            (
                "a reason",
                format!("{HEADER}\nfx\ttests/a\treal\tneed\tc\tmini\n"),
            ),
            ("a row twice", format!("{HEADER}\n{row}\n{row}\n")),
        ] {
            assert!(parse_table(&text).is_err(), "{what} is read");
        }
    }

    /// A row the run does not find and a gap the table does not hold are each one line, a moved
    /// reason both; another pair's rows are not this pair's.
    #[test]
    fn check_rows_names_each_direction() {
        let rows = parse_table(&format!(
            "{HEADER}\nfx\ttests/a\treal\tneed\tkept\tcut\nfx\ttests/a\treal\tneed\tstale\tcut\n\
             fx\ttests/b\treal\tneed\tother\tcut\n"
        ))
        .expect("a well-formed table");
        let found = [
            gap(Side::Real, "kept", Reason::Cut),
            gap(Side::Fixture, "new", Reason::Source),
        ];
        let lines = check_rows(&rows, "fx", "tests/a", &found);
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert!(
            lines[0].contains("holds a row the run does not find") && lines[0].contains("stale")
        );
        assert!(
            lines[1].contains("finds a row the table does not hold") && lines[1].contains("new")
        );
        let moved = [
            gap(Side::Real, "kept", Reason::Source),
            gap(Side::Real, "stale", Reason::Cut),
        ];
        assert_eq!(check_rows(&rows, "fx", "tests/a", &moved).len(), 2);
        assert!(
            check_rows(
                &rows,
                "fx",
                "tests/b",
                &[gap(Side::Real, "other", Reason::Cut)]
            )
            .is_empty()
        );
    }
}
