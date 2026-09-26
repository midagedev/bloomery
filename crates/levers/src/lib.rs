//! bloomery-levers — every `BLOOMERY_*` lever in one table ([`REGISTRY`]),
//! parsed once where a process starts and handed to the engine as typed
//! values.
//!
//! The environment stays the transport: `tools/box.sh` carries
//! `BLOOMERY_BOX_ENV` to the box without a recipe edit. What this crate owns
//! is where a value is parsed. A binary calls [`at_main`] once, first thing in
//! `main`, and hands the typed values to the constructors that use them; the
//! engine does not read the environment for a lever. The exception is the
//! process-wide worker pool, built on first use, which reads its two levers
//! through [`Levers::from_env_named`] — the same parse and the same refusals.
//!
//! A row ([`LeverSpec`]) says what the lever takes ([`Kind`]), what unset
//! means ([`Unset`]), what it is for ([`Class`]) and who reads it ([`Site`]):
//! this crate; a file that still reads it in place, and the round that
//! converts that read; nobody, for a retired name, which is refused whenever
//! it is set; or a runner script. A value the kind does not take is a
//! [`LeverError`] that names the lever, the value and what it takes — never
//! the default.

use std::ffi::{OsStr, OsString};
use std::fmt::{self, Write as _};
use std::path::{Path, PathBuf};

mod registry;
pub use registry::*;

#[cfg(test)]
mod tests;

/// What a lever is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// An operating setting.
    C,
    /// A same-binary A/B arm.
    A,
    /// A twin a gate compares against.
    T,
    /// A debug or instrument switch.
    D,
    /// A mode.
    M,
}

impl Class {
    /// The class's letter and what it stands for.
    #[must_use]
    pub fn describe(self) -> &'static str {
        match self {
            Class::C => "C setting",
            Class::A => "A arm",
            Class::T => "T twin",
            Class::D => "D debug",
            Class::M => "M mode",
        }
    }
}

/// What a lever's value may be. A value is taken as written — no case
/// folding, and whitespace around it ignored only where a [`Kind::Count`]
/// says `trim`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `0` or `1`.
    Flag,
    /// `on` or `off`.
    OnOff,
    /// One of these words.
    Words(&'static [&'static str]),
    /// A whole number from `min` to `max`.
    Count { min: u64, max: u64, trim: bool },
    /// A positive whole multiple of `of`.
    Multiple { of: u64 },
    /// A byte count ([`parse_bytes`]).
    Bytes,
    /// A non-empty path.
    Path,
}

impl Kind {
    /// `v` as a value of this kind, or `None` when the kind does not take it.
    #[must_use]
    pub fn parse(self, v: &str) -> Option<Value> {
        match self {
            Kind::Flag => match v {
                "0" => Some(Value::Flag(false)),
                "1" => Some(Value::Flag(true)),
                _ => None,
            },
            Kind::OnOff => match v {
                "off" => Some(Value::Flag(false)),
                "on" => Some(Value::Flag(true)),
                _ => None,
            },
            Kind::Words(words) => words.iter().find(|&&w| w == v).map(|&w| Value::Word(w)),
            Kind::Count { min, max, trim } => {
                let digits = if trim { v.trim() } else { v };
                digits
                    .parse::<u64>()
                    .ok()
                    .filter(|n| (min..=max).contains(n))
                    .map(Value::Count)
            }
            Kind::Multiple { of } => v
                .parse::<u64>()
                .ok()
                .filter(|&n| n > 0 && n.is_multiple_of(of))
                .map(Value::Count),
            Kind::Bytes => parse_bytes(v).ok().map(Value::Bytes),
            Kind::Path => (!v.is_empty()).then(|| Value::Path(PathBuf::from(v))),
        }
    }

    /// What the kind takes, as a refusal and the tables say it.
    #[must_use]
    pub fn takes(self) -> String {
        match self {
            Kind::Flag => "0 or 1".to_string(),
            Kind::OnOff => "on or off".to_string(),
            Kind::Words(words) => format!("one of {}", words.join(", ")),
            Kind::Count { min, max, trim } => {
                let range = if max == u64::MAX {
                    format!("a whole number from {min} up")
                } else {
                    format!("a whole number from {min} to {max}")
                };
                if trim {
                    format!("{range}, spaces around it ignored")
                } else {
                    range
                }
            }
            Kind::Multiple { of } => format!("a positive multiple of {of}"),
            Kind::Bytes => {
                "bytes, or a whole number of MiB or GiB with an M or G suffix".to_string()
            }
            Kind::Path => "a non-empty path".to_string(),
        }
    }
}

/// A value its kind took.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    /// A [`Kind::Flag`] or [`Kind::OnOff`].
    Flag(bool),
    /// A [`Kind::Words`] word.
    Word(&'static str),
    /// A [`Kind::Count`] or [`Kind::Multiple`] number.
    Count(u64),
    /// A [`Kind::Bytes`] count.
    Bytes(u64),
    /// A [`Kind::Path`].
    Path(PathBuf),
}

/// What an unset lever means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unset {
    /// The value this text is, in the kind's syntax.
    Is(&'static str),
    /// A state of its own, which no value of the kind names.
    Means(&'static str),
}

/// A file that reads a lever in place, not through this crate, and the round
/// that converts or removes that read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InPlace {
    pub file: &'static str,
    pub round: &'static str,
}

/// Who reads a lever.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Site {
    /// This crate parses it: at a binary's `main` ([`at_main`]), or where the
    /// worker pool is built ([`Levers::from_env_named`]). `left` names the
    /// files that still read it in place besides.
    Parsed { left: &'static [InPlace] },
    /// Read in place in the files of `at` only; [`Levers::from_env`] does not
    /// read it.
    Direct { at: &'static [InPlace] },
    /// Not a lever: [`Levers::from_env`] refuses it whenever it is set, with
    /// any value, saying `why`. `left` names the files that refuse it in
    /// place too.
    Retired {
        why: &'static str,
        left: &'static [InPlace],
    },
    /// A runner script's variable; no crate reads it.
    Runner { file: &'static str },
}

impl Site {
    /// Every file that reads the lever in place, with its round.
    #[must_use]
    pub fn in_place(self) -> &'static [InPlace] {
        match self {
            Site::Parsed { left } | Site::Retired { left, .. } => left,
            Site::Direct { at } => at,
            Site::Runner { .. } => &[],
        }
    }

    /// Who reads it, one line.
    #[must_use]
    pub fn describe(self) -> String {
        let places = |p: &[InPlace]| -> String {
            let list: Vec<String> = p
                .iter()
                .map(|i| format!("{} ({})", i.file, i.round))
                .collect();
            list.join(", ")
        };
        match self {
            Site::Parsed { left: [] } => "parsed".to_string(),
            Site::Parsed { left } => format!("parsed; in place in {}", places(left)),
            Site::Direct { at } => format!("in place in {}", places(at)),
            Site::Retired { why, left: [] } => format!("retired: {why}"),
            Site::Retired { why, left } => {
                format!("retired: {why}; refused in place in {}", places(left))
            }
            Site::Runner { file } => format!("runner {file}"),
        }
    }
}

/// One lever's row.
#[derive(Clone, Copy, Debug)]
pub struct LeverSpec {
    /// The environment variable.
    pub name: &'static str,
    pub class: Class,
    pub kind: Kind,
    /// What unset means.
    pub default: Unset,
    /// What it does, and where a binary prints it.
    pub doc: &'static str,
    pub site: Site,
}

impl LeverSpec {
    /// The value unset stands for; `None` when unset is a state of its own.
    #[must_use]
    pub fn default_value(&self) -> Option<Value> {
        match self.default {
            Unset::Is(text) => Some(self.kind.parse(text).unwrap_or_else(|| {
                panic!(
                    "{}: the registry's default {text:?} is not a value of its kind",
                    self.name
                )
            })),
            Unset::Means(_) => None,
        }
    }
}

/// The row of `name`, if it is a lever of [`REGISTRY`].
#[must_use]
pub fn spec(name: &str) -> Option<&'static LeverSpec> {
    REGISTRY.iter().find(|r| r.name == name)
}

/// A lever set to a value its kind does not take, or a retired name that is
/// set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeverError {
    /// The lever.
    pub name: &'static str,
    /// The value as set; lossily, when it is not UTF-8.
    pub value: String,
    /// What the lever takes, or why it is no lever.
    pub expected: String,
}

impl fmt::Display for LeverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}={:?} is refused: {}",
            self.name, self.value, self.expected
        )
    }
}

impl std::error::Error for LeverError {}

/// One Parsed lever as a reading found it.
#[derive(Clone, Debug)]
struct Entry {
    row: &'static LeverSpec,
    /// The text as set; `None` unset.
    set: Option<String>,
    /// The value: as set, else the default; `None` unset when unset is a
    /// state of its own ([`Unset::Means`]).
    value: Option<Value>,
}

impl Entry {
    fn read(row: &'static LeverSpec, set: Option<OsString>) -> Result<Entry, LeverError> {
        let refuse = |value: String, prefix: &str| LeverError {
            name: row.name,
            value,
            expected: format!("{prefix}it takes {}", row.kind.takes()),
        };
        let Some(raw) = set else {
            return Ok(Entry {
                row,
                set: None,
                value: row.default_value(),
            });
        };
        let text = raw
            .into_string()
            .map_err(|raw| refuse(raw.to_string_lossy().into_owned(), "not UTF-8; "))?;
        let value = row
            .kind
            .parse(&text)
            .ok_or_else(|| refuse(text.clone(), ""))?;
        Ok(Entry {
            row,
            set: Some(text),
            value: Some(value),
        })
    }
}

/// The Parsed levers one reading found, each parsed by its kind.
#[derive(Clone, Debug)]
pub struct Levers {
    entries: Vec<Entry>,
}

impl Levers {
    /// Every Parsed lever of [`REGISTRY`] from the process environment, each
    /// read once and parsed by its kind. Refused when one is set to a value
    /// its kind does not take (or one that is not UTF-8), and when a retired
    /// name is set. The environment's other variables are not read.
    pub fn from_env() -> Result<Levers, LeverError> {
        Levers::read(REGISTRY.iter(), |name| std::env::var_os(name))
    }

    /// [`Levers::from_env`] over `pairs` instead of the environment: a name
    /// given twice takes its last value, and a name that is no lever is
    /// ignored, as the environment's other variables are.
    pub fn from_pairs<V: AsRef<OsStr>>(pairs: &[(&str, V)]) -> Result<Levers, LeverError> {
        Levers::read(REGISTRY.iter(), |name| {
            pairs
                .iter()
                .rev()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| v.as_ref().to_os_string())
        })
    }

    /// Only the Parsed levers `names`, from the process environment: for a
    /// process-wide reader no `main` hands a value to (the worker pool). A
    /// name that is not a Parsed lever panics by name — a caller's mistake.
    pub fn from_env_named(names: &[&str]) -> Result<Levers, LeverError> {
        let rows = names.iter().map(|&n| match spec(n) {
            Some(row) if matches!(row.site, Site::Parsed { .. }) => row,
            _ => panic!("{n} is not a lever this crate parses"),
        });
        Levers::read(rows, |name| std::env::var_os(name))
    }

    fn read(
        rows: impl Iterator<Item = &'static LeverSpec>,
        lookup: impl Fn(&str) -> Option<OsString>,
    ) -> Result<Levers, LeverError> {
        let mut entries = Vec::new();
        for row in rows {
            match row.site {
                Site::Parsed { .. } => entries.push(Entry::read(row, lookup(row.name))?),
                Site::Retired { why, .. } => {
                    if let Some(raw) = lookup(row.name) {
                        return Err(LeverError {
                            name: row.name,
                            value: raw.to_string_lossy().into_owned(),
                            expected: format!("it is no lever: {why}; unset it"),
                        });
                    }
                }
                Site::Direct { .. } | Site::Runner { .. } => {}
            }
        }
        Ok(Levers { entries })
    }

    fn entry(&self, name: &str) -> &Entry {
        self.entries
            .iter()
            .find(|e| e.row.name == name)
            .unwrap_or_else(|| panic!("{name} is not among the levers this reading parsed"))
    }

    fn flag(&self, name: &str) -> bool {
        match self.entry(name).value {
            Some(Value::Flag(b)) => b,
            ref v => panic!("{name} holds {v:?}, not a flag with a default"),
        }
    }

    fn word(&self, name: &str) -> Option<&'static str> {
        match self.entry(name).value {
            None => None,
            Some(Value::Word(w)) => Some(w),
            ref v => panic!("{name} holds {v:?}, not a word"),
        }
    }

    fn count(&self, name: &str) -> Option<u64> {
        match self.entry(name).value {
            None => None,
            Some(Value::Count(n)) => Some(n),
            ref v => panic!("{name} holds {v:?}, not a count"),
        }
    }

    fn usize_of(name: &str, n: u64) -> usize {
        usize::try_from(n).unwrap_or_else(|_| panic!("{name}={n} passes usize"))
    }

    fn defaulted<T>(name: &str, v: Option<T>) -> T {
        v.unwrap_or_else(|| panic!("{name} has no default in the registry"))
    }

    /// `BLOOMERY_THREADS`: the pool's thread count; `None` unset (the
    /// physical core count).
    #[must_use]
    pub fn threads(&self) -> Option<usize> {
        self.count(THREADS).map(|n| Levers::usize_of(THREADS, n))
    }

    /// `BLOOMERY_SPIN`: a waiting pool thread's spin iterations.
    #[must_use]
    pub fn spin(&self) -> u64 {
        Levers::defaulted(SPIN, self.count(SPIN))
    }

    /// `BLOOMERY_CED`: the V4.1 prompt call runs the CED triangle.
    #[must_use]
    pub fn ced(&self) -> bool {
        self.flag(CED)
    }

    /// `BLOOMERY_PREFILL`: how a V4.1 binary feeds a prompt, `batch` or
    /// `steps`.
    #[must_use]
    pub fn prefill(&self) -> &'static str {
        Levers::defaulted(PREFILL, self.word(PREFILL))
    }

    /// `BLOOMERY_PREFILL_GROUP`: batches a V4.1 prompt group holds.
    #[must_use]
    pub fn prefill_group(&self) -> usize {
        let n = Levers::defaulted(PREFILL_GROUP, self.count(PREFILL_GROUP));
        Levers::usize_of(PREFILL_GROUP, n)
    }

    /// `BLOOMERY_ENGRAM_HELPER`: V4.1 step rows read by a helper thread.
    #[must_use]
    pub fn engram_helper(&self) -> bool {
        self.flag(ENGRAM_HELPER)
    }

    /// `BLOOMERY_STEP_STATS`: the step statistics are read and printed.
    #[must_use]
    pub fn step_stats(&self) -> bool {
        self.flag(STEP_STATS)
    }

    /// `BLOOMERY_HOT_LIST`: the hot list file; `None` unset (the id prefix).
    #[must_use]
    pub fn hot_list(&self) -> Option<&Path> {
        match &self.entry(HOT_LIST).value {
            None => None,
            Some(Value::Path(p)) => Some(p),
            v => panic!("{HOT_LIST} holds {v:?}, not a path"),
        }
    }

    /// `BLOOMERY_CARD_BUDGET`: a card byte budget; `None` unset.
    #[must_use]
    pub fn card_budget(&self) -> Option<u64> {
        match self.entry(CARD_BUDGET).value {
            None => None,
            Some(Value::Bytes(b)) => Some(b),
            ref v => panic!("{CARD_BUDGET} holds {v:?}, not bytes"),
        }
    }

    /// `BLOOMERY_PIN_MAIN`: a binary pins its main thread.
    #[must_use]
    pub fn pin_main(&self) -> bool {
        self.flag(PIN_MAIN)
    }

    /// `BLOOMERY_DRAFT`: the served draft, `lookup` or `dspark`; `None`
    /// unset (the plain path).
    #[must_use]
    pub fn draft(&self) -> Option<&'static str> {
        self.word(DRAFT)
    }

    /// `BLOOMERY_CHECK_FINITE`: the finite probe runs.
    #[must_use]
    pub fn check_finite(&self) -> bool {
        self.flag(CHECK_FINITE)
    }

    /// Every row of [`REGISTRY`], one line each: the value this reading has
    /// — as set, the default, or `-` for a lever it does not parse — and
    /// who reads it. What `--levers` prints ([`at_main`]).
    #[must_use]
    pub fn table(&self) -> String {
        let mut out = String::new();
        for row in REGISTRY {
            let value = match self.entries.iter().find(|e| std::ptr::eq(e.row, row)) {
                Some(Entry {
                    set: Some(text), ..
                }) => format!("set {text}"),
                Some(Entry { set: None, .. }) => match row.default {
                    Unset::Is(d) => format!("default {d}"),
                    Unset::Means(m) => format!("unset: {m}"),
                },
                None => "-".to_string(),
            };
            let _ = writeln!(
                out,
                "{:<26} {:<9} {:<32} {}",
                row.name,
                row.class.describe(),
                value,
                row.site.describe()
            );
        }
        out
    }
}

/// A binary's one reading of the levers, first thing in `main`:
/// [`Levers::from_env`]. With `--levers` among the process's arguments it
/// prints [`Levers::table`] on stdout and exits 0 instead.
pub fn at_main() -> Result<Levers, LeverError> {
    let levers = Levers::from_env()?;
    if std::env::args_os().skip(1).any(|a| a == "--levers") {
        print!("{}", levers.table());
        std::process::exit(0);
    }
    Ok(levers)
}

/// [`REGISTRY`] as a Markdown table — lever, class, what it takes, what
/// unset means, who reads it, what it does: the repository's lever
/// documentation.
#[must_use]
pub fn markdown() -> String {
    let mut out = String::from(
        "| lever | class | takes | unset | read by | what it does |\n|---|---|---|---|---|---|\n",
    );
    for row in REGISTRY {
        let unset = match row.default {
            Unset::Is(d) => format!("`{d}`"),
            Unset::Means(m) => m.to_string(),
        };
        let _ = writeln!(
            out,
            "| `{}` | {} | {} | {} | {} | {} |",
            row.name,
            row.class.describe(),
            row.kind.takes(),
            unset,
            row.site.describe(),
            row.doc
        );
    }
    out
}

/// Bytes `value` names: digits with optional `_` separators, then nothing
/// (bytes), `M` (MiB) or `G` (GiB). Anything else, and a value that passes
/// `u64`, is refused with the reason.
pub fn parse_bytes(value: &str) -> Result<u64, String> {
    let (digits, unit) = match value.strip_suffix('G') {
        Some(d) => (d, 1u64 << 30),
        None => match value.strip_suffix('M') {
            Some(d) => (d, 1u64 << 20),
            None => (value, 1),
        },
    };
    let digits: String = digits.chars().filter(|&c| c != '_').collect();
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!(
            "{value:?} is not bytes, or a count of MiB or GiB with an M or G suffix"
        ));
    }
    digits
        .parse::<u64>()
        .ok()
        .and_then(|n| n.checked_mul(unit))
        .ok_or_else(|| format!("{value:?} passes u64 bytes"))
}
