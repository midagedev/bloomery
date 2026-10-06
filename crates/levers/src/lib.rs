//! bloomery-levers — every `BLOOMERY_*` name in one table, the registry: the
//! levers, parsed once where a process starts and handed to the engine as
//! typed values, and the names that are no lever — a path, a runner's
//! setting, a handshake between processes — which a process's environment
//! may carry.
//!
//! The environment stays the transport: `tools/box.sh` carries
//! `BLOOMERY_BOX_ENV` to the box without a recipe edit. What this crate owns
//! is where a value is parsed. A binary calls [`at_main`] once, first thing in
//! `main`, with the levers its run acts on, and hands the typed values to the
//! constructors that use them; the engine does not read the environment for
//! a lever. The exception is the process-wide worker pool, built on first
//! use, which reads its two levers through [`pool_levers`]: the parse
//! [`at_main`] runs for those two and the refusal of a retired name, nothing
//! else — which levers a binary acts on, and which names it knows, is its
//! `main`'s reading.
//!
//! A row says what the name takes, what unset means, what it is for and who
//! reads it: this crate; a file that still reads it in place, and the round
//! that converts that read; nobody, for a retired name, which every reading
//! here refuses when it is set; or, for a name that is no lever, the script
//! or harness that owns it. A value the kind does not take is a refusal that
//! names the lever, the value and what it takes — never the default.

use std::ffi::{OsStr, OsString};
use std::fmt::{self, Write as _};
use std::path::{Path, PathBuf};

mod registry;
use registry::REGISTRY;
pub use registry::{
    CARD_BUDGET, CARD_DONTNEED, CED, CHECK_FINITE, DRAFT, ENGRAM_HELPER, GEN_SLOTS, GEN_SLOTS_MAX,
    HOST_LOCK, HOST_POPULATE, HOSTSTREAM, LANE_PREFETCH, LANE_PREFETCH_DEFAULT, MTP_DRAFT,
    MTP_HEAD_ROWS, MTP_WINDOWS, PIN_MAIN, PREFILL, PREFILL_GROUP, PREFILL_GROUP_DEFAULT,
    PREFILL_GROUP_MAX, QWEN3_KV, QWEN38_EXPERTS, R8, RESIDENCY, ROUTE_TRACE, SPIN, STEP_STATS,
    THREADS,
};

#[cfg(test)]
mod tests;

/// What a name is for: a lever's use, or what a name that is no lever names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Class {
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
    /// No lever: where a file or a directory is.
    P,
    /// No lever: a runner's or a harness's own variable — a bound, a card, a
    /// lease, a handshake between two processes.
    R,
}

impl Class {
    /// The class's letter and what it stands for.
    pub(crate) fn describe(self) -> &'static str {
        match self {
            Class::C => "C setting",
            Class::A => "A arm",
            Class::T => "T twin",
            Class::D => "D debug",
            Class::M => "M mode",
            Class::P => "P path",
            Class::R => "R runner",
        }
    }
}

/// What a name's value may be. A value is taken as written: no case folding,
/// whitespace around it ignored only where a [`Kind::Count`] says `trim`, and
/// a number in base 10 with no sign and no leading zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
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
    /// One of these words, or else a non-empty path: a word is never read
    /// as a path (a file of that name is named `./<word>`).
    PathOr(&'static [&'static str]),
    /// The path of a regular file that exists when the value is read.
    File,
    /// What the name's owner takes: a name no reading here parses.
    Text,
    /// `off`, or the adaptive residency rule's `mid-p<P>-s<S>`
    /// ([`residency_word`]).
    Residency,
}

/// Why [`whole`] refused a number.
enum NotWhole {
    /// Not base-10 digits, or a leading zero.
    Syntax,
    /// More than a `u64` holds.
    Overflow,
}

/// `s` as a whole number: base-10 digits with no sign and no leading zero.
fn whole(s: &str) -> Result<u64, NotWhole> {
    let digits = !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if !digits || (s.len() > 1 && s.starts_with('0')) {
        return Err(NotWhole::Syntax);
    }
    s.parse::<u64>().map_err(|_| NotWhole::Overflow)
}

/// What a number past `u64` is refused for.
const OVERFLOW: &str = "more than a u64 holds";

impl Kind {
    /// `v` as a value of this kind, or the refusal's text: what the kind
    /// takes, after the reason when that alone does not say why.
    pub(crate) fn parse(self, v: &str) -> Result<Value, String> {
        let refused = |reason: Option<&str>| match reason {
            Some(r) => format!("{r}; it takes {}", self.takes()),
            None => format!("it takes {}", self.takes()),
        };
        let number = |n: Result<u64, NotWhole>, fits: &dyn Fn(u64) -> bool| match n {
            Ok(n) if fits(n) => Ok(Value::Count(n)),
            Ok(_) | Err(NotWhole::Syntax) => Err(refused(None)),
            Err(NotWhole::Overflow) => Err(refused(Some(OVERFLOW))),
        };
        match self {
            Kind::Flag => match v {
                "0" => Ok(Value::Flag(false)),
                "1" => Ok(Value::Flag(true)),
                _ => Err(refused(None)),
            },
            Kind::OnOff => match v {
                "off" => Ok(Value::Flag(false)),
                "on" => Ok(Value::Flag(true)),
                _ => Err(refused(None)),
            },
            Kind::Words(words) => words
                .iter()
                .find(|&&w| w == v)
                .map(|&w| Value::Word(w))
                .ok_or_else(|| refused(None)),
            Kind::Count { min, max, trim } => {
                let digits = if trim { v.trim() } else { v };
                number(whole(digits), &|n| (min..=max).contains(&n))
            }
            Kind::Multiple { of } => number(whole(v), &|n| n > 0 && n.is_multiple_of(of)),
            Kind::Bytes => parse_bytes(v).map(Value::Bytes).map_err(|e| match e {
                BytesError::NotBytes(_) => refused(None),
                BytesError::Overflow(_) => refused(Some(e.reason())),
            }),
            Kind::Path if !v.is_empty() => Ok(Value::Path(PathBuf::from(v))),
            Kind::PathOr(words) => match words.iter().find(|&&w| w == v) {
                Some(&w) => Ok(Value::Word(w)),
                None if !v.is_empty() => Ok(Value::Path(PathBuf::from(v))),
                None => Err(refused(None)),
            },
            Kind::File if Path::new(v).is_file() => Ok(Value::Path(PathBuf::from(v))),
            Kind::File if !v.is_empty() => Err(refused(Some("no regular file is at that path"))),
            Kind::Residency => match residency_word(v) {
                Some(ResidencyWord::Off) => Ok(Value::Word("off")),
                // Parsed once at `main`: the word lives for the process, as
                // a listed word would.
                Some(ResidencyWord::Mid { .. }) => {
                    Ok(Value::Word(Box::leak(v.to_owned().into_boxed_str())))
                }
                None => Err(refused(None)),
            },
            Kind::Path | Kind::File | Kind::Text => Err(refused(None)),
        }
    }

    /// What the kind takes, as a refusal and the tables say it.
    pub(crate) fn takes(self) -> String {
        const DIGITS: &str = "base 10, no sign or leading zero";
        match self {
            Kind::Flag => "0 or 1".to_string(),
            Kind::OnOff => "on or off".to_string(),
            Kind::Words(words) => format!("one of {}", words.join(", ")),
            Kind::Count { min, max, trim } => {
                let range = if max == u64::MAX {
                    format!("a whole number from {min} up ({DIGITS})")
                } else {
                    format!("a whole number from {min} to {max} ({DIGITS})")
                };
                if trim {
                    format!("{range}, spaces around it ignored")
                } else {
                    range
                }
            }
            Kind::Multiple { of } => format!("a positive multiple of {of} ({DIGITS})"),
            Kind::Bytes => {
                "bytes, or a whole number of MiB or GiB with an M or G suffix".to_string()
            }
            Kind::Path => "a non-empty path".to_string(),
            Kind::PathOr(words) => format!("{}, or a non-empty path", words.join(", ")),
            Kind::File => "the path of an existing regular file".to_string(),
            Kind::Text => "what its owner takes".to_string(),
            Kind::Residency => {
                format!("off, or mid-p<P>-s<S> with P and S whole numbers ({DIGITS}), S at least 1")
            }
        }
    }
}

/// A [`Kind::Residency`] value: what `BLOOMERY_RESIDENCY` asks a load for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidencyWord {
    /// No machine: the load's slot map for the model's life.
    Off,
    /// The rule's `mid` parameters: `pinned` seed experts a layer never a
    /// victim, `spares` free slots a layer.
    Mid { pinned: usize, spares: usize },
}

/// `v` in the residency lever's grammar, the one owner of it: `off`, or
/// `mid-p<P>-s<S>` with `P` and `S` whole numbers in base 10 with no sign or
/// leading zero and `S` at least 1; `None` for anything else.
#[must_use]
pub fn residency_word(v: &str) -> Option<ResidencyWord> {
    if v == "off" {
        return Some(ResidencyWord::Off);
    }
    let (p, s) = v.strip_prefix("mid-p")?.split_once("-s")?;
    let num = |t: &str| usize::try_from(whole(t).ok()?).ok();
    match (num(p)?, num(s)?) {
        (pinned, spares) if spares >= 1 => Some(ResidencyWord::Mid { pinned, spares }),
        _ => None,
    }
}

/// A set `BLOOMERY_MTP_HEAD_ROWS`: the head the MTP draft scores with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtpHead<'a> {
    /// `full`: every token of the vocabulary.
    Full,
    /// The row list at this path.
    List(&'a Path),
}

/// A value its kind took.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Value {
    /// A [`Kind::Flag`] or [`Kind::OnOff`].
    Flag(bool),
    /// A [`Kind::Words`] or [`Kind::PathOr`] word.
    Word(&'static str),
    /// A [`Kind::Count`] or [`Kind::Multiple`] number.
    Count(u64),
    /// A [`Kind::Bytes`] count.
    Bytes(u64),
    /// A [`Kind::Path`], [`Kind::PathOr`] or [`Kind::File`] path.
    Path(PathBuf),
}

/// What an unset name means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Unset {
    /// The value this text is, in the kind's syntax.
    Is(&'static str),
    /// A state of its own, which no value of the kind names.
    Means(&'static str),
}

/// A file that reads a lever in place, not through this crate, and the round
/// that converts or removes that read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct InPlace {
    /// The file, from the repository root: a line of `tools/levers-direct.txt`.
    pub(crate) file: &'static str,
    /// The round that converts or removes the read: that line's third column.
    pub(crate) round: &'static str,
}

/// Who reads a name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Site {
    /// This crate parses it: at a binary's `main` ([`at_main`]), or where the
    /// worker pool is built ([`pool_levers`]). `left` names the files that
    /// still read it in place besides.
    Parsed { left: &'static [InPlace] },
    /// Read in place in the files of `at` only; no reading here parses it.
    Direct { at: &'static [InPlace] },
    /// Not a lever: every reading here refuses it when it is set, with any
    /// value, saying `why`. `left` names the files that refuse it in place
    /// too.
    Retired {
        why: &'static str,
        left: &'static [InPlace],
    },
    /// No lever: a name a script under `tools/` sets or reads, or, with no
    /// `script`, a harness does. No reading here parses it and each knows
    /// it, so a process started with it set runs. A crate's read of it is a
    /// line of `tools/levers-direct.txt`, which names the file.
    Env {
        /// The script's path relative to `tools/` — a name, which
        /// `tools/recipes.py` does not take for an input of every binary
        /// that links this crate, as it takes a whole repository path.
        script: Option<&'static str>,
    },
}

impl Site {
    /// Who reads it, one line.
    pub(crate) fn describe(self) -> String {
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
            Site::Env {
                script: Some(script),
            } => format!("tools/{script}"),
            Site::Env { script: None } => "a harness, in place".to_string(),
        }
    }
}

/// One name's row.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LeverSpec {
    /// The environment variable.
    pub(crate) name: &'static str,
    /// What it is for.
    pub(crate) class: Class,
    /// What its value may be.
    pub(crate) kind: Kind,
    /// What unset means.
    pub(crate) default: Unset,
    /// What it does, and where a binary prints it.
    pub(crate) doc: &'static str,
    /// Who reads it.
    pub(crate) site: Site,
}

impl LeverSpec {
    /// The value unset stands for; `None` when unset is a state of its own.
    pub(crate) fn default_value(&self) -> Option<Value> {
        match self.default {
            Unset::Is(text) => Some(self.kind.parse(text).unwrap_or_else(|e| {
                panic!(
                    "{}: the registry's default {text:?} is not a value of its kind: {e}",
                    self.name
                )
            })),
            Unset::Means(_) => None,
        }
    }

    /// Whether a person sets it to change what a binary does: not a path's
    /// or a runner's own name.
    pub(crate) fn is_lever(&self) -> bool {
        !matches!(self.site, Site::Env { .. })
    }
}

/// The row of `name`, if [`REGISTRY`] has one.
pub(crate) fn spec(name: &str) -> Option<&'static LeverSpec> {
    REGISTRY.iter().find(|r| r.name == name)
}

/// Why a reading refused a variable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Why {
    /// A value the lever's kind does not take: what it takes, after the
    /// reason when that alone does not say why.
    Value(String),
    /// A retired name: why it is no lever.
    Retired(&'static str),
    /// A lever the binary this names does not act on.
    NotActedOn(String),
    /// A `BLOOMERY_*` name no row names, and the row name nearest to it.
    Unknown(&'static str),
}

/// One variable a reading refused: its name, its value as set (lossily, when
/// it is not UTF-8), and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Refusal {
    pub(crate) name: String,
    pub(crate) value: String,
    pub(crate) why: Why,
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={:?} is refused: ", self.name, self.value)?;
        match &self.why {
            Why::Value(expected) => write!(f, "{expected}"),
            Why::Retired(why) => write!(f, "it is no lever: {why}; unset it"),
            Why::NotActedOn(bin) => {
                write!(f, "this binary ({bin}) does not act on it; unset it")
            }
            Why::Unknown(nearest) => write!(
                f,
                "no row of the lever registry names it (the nearest is {nearest}); unset it or \
                 fix the name"
            ),
        }
    }
}

/// Every variable one reading refused: the levers and retired names in the
/// registry's order, then the names no row names. Its text is a line each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeverError {
    refused: Vec<Refusal>,
}

impl fmt::Display for LeverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, r) in self.refused.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{r}")?;
        }
        Ok(())
    }
}

impl std::error::Error for LeverError {}

/// `v`, lossily, when it is not UTF-8.
fn lossy(v: &OsStr) -> String {
    v.to_string_lossy().into_owned()
}

/// One Parsed lever as a reading found it.
#[derive(Clone, Debug)]
struct Entry {
    row: &'static LeverSpec,
    /// Whether the reading's binary acts on it; one it does not is unset
    /// (set, it was refused) and holds the default.
    acts: bool,
    /// The text as set; `None` unset.
    set: Option<String>,
    /// The value: as set, else the default; `None` unset when unset is a
    /// state of its own ([`Unset::Means`]).
    value: Option<Value>,
}

impl Entry {
    fn read(row: &'static LeverSpec, raw: Option<&OsStr>, acts: bool) -> Result<Entry, Refusal> {
        let refuse = |value: String, expected: String| Refusal {
            name: row.name.to_string(),
            value,
            why: Why::Value(expected),
        };
        let Some(raw) = raw else {
            return Ok(Entry {
                row,
                acts,
                set: None,
                value: row.default_value(),
            });
        };
        let Some(text) = raw.to_str() else {
            let expected = format!("not UTF-8; it takes {}", row.kind.takes());
            return Err(refuse(lossy(raw), expected));
        };
        let value = row
            .kind
            .parse(text)
            .map_err(|expected| refuse(text.to_string(), expected))?;
        Ok(Entry {
            row,
            acts,
            set: Some(text.to_string()),
            value: Some(value),
        })
    }
}

/// Whose reading it is, which fixes what it refuses besides a value a
/// lever's kind does not take and a retired name that is set.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Scope<'a> {
    /// A binary's `main` ([`at_main`]): a Parsed lever set that is not in
    /// `acts_on` — the pool's two always are — and a `BLOOMERY_*` name no
    /// row names.
    Main {
        bin: &'a str,
        acts_on: &'a [&'a str],
    },
    /// A harness that acts on every Parsed lever ([`Levers::from_env`]):
    /// nothing more.
    Every,
    /// The worker pool ([`pool_levers`]): it reads its two levers alone; the
    /// others are its binary's to read.
    Pool,
}

/// The levers of `scope` over `env`, every name once — the last of a name
/// given twice — and every refusal at once.
pub(crate) fn read(env: &[(OsString, OsString)], scope: Scope<'_>) -> Result<Levers, LeverError> {
    if let Scope::Main { acts_on, .. } = scope
        && let Some(name) = not_parsed(acts_on)
    {
        panic!("{name} is not a lever this crate parses");
    }
    let get = |name: &str| {
        env.iter()
            .rev()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_os_str())
    };
    let pool = |name: &str| name == THREADS || name == SPIN;
    let mut entries = Vec::new();
    let mut refused = Vec::new();
    for row in REGISTRY {
        match row.site {
            Site::Parsed { .. } => {
                let acts = match scope {
                    Scope::Main { acts_on, .. } => pool(row.name) || acts_on.contains(&row.name),
                    Scope::Every => true,
                    Scope::Pool if pool(row.name) => true,
                    Scope::Pool => continue,
                };
                match (get(row.name), scope) {
                    (Some(v), Scope::Main { bin, .. }) if !acts => refused.push(Refusal {
                        name: row.name.to_string(),
                        value: lossy(v),
                        why: Why::NotActedOn(bin.to_string()),
                    }),
                    (raw, _) => match Entry::read(row, raw, acts) {
                        Ok(e) => entries.push(e),
                        Err(r) => refused.push(r),
                    },
                }
            }
            Site::Retired { why, .. } => {
                if let Some(v) = get(row.name) {
                    refused.push(Refusal {
                        name: row.name.to_string(),
                        value: lossy(v),
                        why: Why::Retired(why),
                    });
                }
            }
            Site::Direct { .. } | Site::Env { .. } => {}
        }
    }
    if let Scope::Main { .. } = scope {
        let mut seen: Vec<&OsStr> = Vec::new();
        for (k, _) in env {
            let known = k.to_str().is_some_and(|n| spec(n).is_some());
            if !k.as_encoded_bytes().starts_with(b"BLOOMERY_") || known || seen.contains(&&**k) {
                continue;
            }
            seen.push(k.as_os_str());
            let last = env.iter().rev().find(|(name, _)| name == k);
            let name = lossy(k);
            refused.push(Refusal {
                why: Why::Unknown(nearest(&name)),
                name,
                value: last.map_or_else(String::new, |(_, v)| lossy(v)),
            });
        }
    }
    if refused.is_empty() {
        Ok(Levers { entries })
    } else {
        Err(LeverError { refused })
    }
}

/// The first of `names` that is not a Parsed lever's.
fn not_parsed<'a>(names: &[&'a str]) -> Option<&'a str> {
    names
        .iter()
        .copied()
        .find(|&name| !spec(name).is_some_and(|r| matches!(r.site, Site::Parsed { .. })))
}

/// The row name nearest `name` by edits (an insertion, a deletion or a
/// substitution of one byte each); the first in the registry's order on a
/// tie.
fn nearest(name: &str) -> &'static str {
    let edits = |a: &[u8], b: &[u8]| -> usize {
        let mut row: Vec<usize> = (0..=b.len()).collect();
        for (i, &ca) in a.iter().enumerate() {
            let mut diag = row[0];
            row[0] = i + 1;
            for (j, &cb) in b.iter().enumerate() {
                let next = (diag + usize::from(ca != cb))
                    .min(row[j] + 1)
                    .min(row[j + 1] + 1);
                diag = row[j + 1];
                row[j + 1] = next;
            }
        }
        row[b.len()]
    };
    REGISTRY
        .iter()
        .min_by_key(|r| edits(name.as_bytes(), r.name.as_bytes()))
        .map_or("", |r| r.name)
}

/// The Parsed levers one reading found, each parsed by its kind.
#[derive(Clone, Debug)]
pub struct Levers {
    entries: Vec<Entry>,
}

impl Levers {
    /// Every Parsed lever of the registry from the process environment, for a
    /// harness that acts on all of them (a test that plans under the
    /// placement's levers): each parsed by its kind. Refused, with every
    /// refusal at once, when one is set to a value its kind does not take
    /// (or one that is not UTF-8), and when a retired name is set. A binary
    /// reads with [`at_main`] instead.
    pub fn from_env() -> Result<Levers, LeverError> {
        let env: Vec<(OsString, OsString)> = std::env::vars_os().collect();
        read(&env, Scope::Every)
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

    /// `BLOOMERY_PREFILL_GROUP`: batches a V4.1 or GLM-5.3 prompt group
    /// holds, 1 to [`PREFILL_GROUP_MAX`].
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

    /// `BLOOMERY_CARD_BUDGET`: a card budget in bytes; `None` unset.
    #[must_use]
    pub fn card_budget_bytes(&self) -> Option<u64> {
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

    /// `BLOOMERY_DRAFT`: the served draft, `lookup`, `dspark` or `mtp`; `None`
    /// unset (the plain path).
    #[must_use]
    pub fn draft(&self) -> Option<&'static str> {
        self.word(DRAFT)
    }

    /// `BLOOMERY_MTP_HEAD_ROWS`: the MTP draft's head, `full` or a row
    /// list's path; `None` unset (the shipped list for its tokenizer's
    /// target, the full head for any other: the model crate's rule).
    #[must_use]
    pub fn mtp_head_rows(&self) -> Option<MtpHead<'_>> {
        match &self.entry(MTP_HEAD_ROWS).value {
            None => None,
            Some(Value::Word("full")) => Some(MtpHead::Full),
            Some(Value::Path(p)) => Some(MtpHead::List(p)),
            v => panic!("{MTP_HEAD_ROWS} holds {v:?}, not full or a path"),
        }
    }

    /// `BLOOMERY_MTP_DRAFT`: the MTP draft file, which existed when the
    /// reading ran; `None` unset (the shared draft beside the target, else
    /// the family's path — `refset::arch::qwen4exp::mtp::draft_file`).
    #[must_use]
    pub fn mtp_draft(&self) -> Option<&Path> {
        match &self.entry(MTP_DRAFT).value {
            None => None,
            Some(Value::Path(p)) => Some(p),
            v => panic!("{MTP_DRAFT} holds {v:?}, not a path"),
        }
    }

    /// `BLOOMERY_MTP_WINDOWS`: every drafted window's proposal, probabilities
    /// and kept count are printed.
    #[must_use]
    pub fn mtp_windows(&self) -> bool {
        self.flag(MTP_WINDOWS)
    }

    /// `BLOOMERY_ROUTE_TRACE`: the route trace's new directory; `None` unset
    /// (no trace).
    #[must_use]
    pub fn route_trace(&self) -> Option<&Path> {
        match &self.entry(ROUTE_TRACE).value {
            None => None,
            Some(Value::Path(p)) => Some(p),
            v => panic!("{ROUTE_TRACE} holds {v:?}, not a path"),
        }
    }

    /// `BLOOMERY_QWEN38_EXPERTS`: where a Qwen3.8 plan puts the routed
    /// experts, `host` or `card` — as set, else [`QWEN38_EXPERTS_UNSET`].
    #[must_use]
    pub fn qwen38_experts(&self) -> &'static str {
        self.qwen38_experts_set().unwrap_or(QWEN38_EXPERTS_UNSET)
    }

    /// `BLOOMERY_QWEN38_EXPERTS` as set; `None` unset, which another
    /// family's file reads as nothing to refuse.
    #[must_use]
    pub fn qwen38_experts_set(&self) -> Option<&'static str> {
        self.word(QWEN38_EXPERTS)
    }

    /// `BLOOMERY_QWEN3_KV` as set: the K/V planes' format, `f16` or `q8_0`;
    /// `None` unset, which the load reads as the registry's `f16` default.
    #[must_use]
    pub fn qwen3_kv_set(&self) -> Option<&'static str> {
        self.word(QWEN3_KV)
    }

    /// `BLOOMERY_CHECK_FINITE`: the finite probe runs.
    #[must_use]
    pub fn check_finite(&self) -> bool {
        self.flag(CHECK_FINITE)
    }

    /// `BLOOMERY_RESIDENCY` as set: `off`, or the adaptive residency rule's
    /// word, whose grammar is [`residency_word`];
    /// `None` unset, which [`Levers::residency_at`] resolves.
    #[must_use]
    pub fn residency(&self) -> Option<&'static str> {
        self.word(RESIDENCY)
    }

    /// `BLOOMERY_RESIDENCY` as a load at `at` runs it: as set, else what
    /// [`residency_unset`] picks there.
    #[must_use]
    pub fn residency_at(&self, at: ResidencyAt) -> ResidencyPick {
        match self.residency() {
            Some(word) => ResidencyPick {
                word,
                why: ResidencyWhy::Set,
            },
            None => residency_unset(at),
        }
    }

    /// `BLOOMERY_HOSTSTREAM`: a V4.1 prompt call streams its hottest host
    /// experts into the residency's pool; `None` when unset, which follows
    /// the residency (the body resolves it).
    #[must_use]
    pub fn hoststream(&self) -> Option<bool> {
        match self.entry(HOSTSTREAM).value {
            None => None,
            Some(Value::Flag(b)) => Some(b),
            ref v => panic!("{HOSTSTREAM} holds {v:?}, not a flag"),
        }
    }

    /// `BLOOMERY_LANE_PREFETCH`: the host union's row-lane packs prefetch
    /// each group's successor.
    #[must_use]
    pub fn lane_prefetch(&self) -> bool {
        self.flag(LANE_PREFETCH)
    }

    /// `BLOOMERY_GEN_SLOTS`: the streams `generate_qwen3moe` decodes in one
    /// pass, 1 to [`GEN_SLOTS_MAX`].
    #[must_use]
    pub fn gen_slots(&self) -> usize {
        let n = Levers::defaulted(GEN_SLOTS, self.count(GEN_SLOTS));
        Levers::usize_of(GEN_SLOTS, n)
    }

    /// The host tier's load settings: [`HOST_POPULATE`], [`HOST_LOCK`],
    /// [`CARD_DONTNEED`] and [`R8`].
    #[must_use]
    pub fn host(&self) -> HostCfg {
        HostCfg {
            populate: self.flag(HOST_POPULATE),
            lock: self.flag(HOST_LOCK),
            card_dontneed: self.flag(CARD_DONTNEED),
            r8: self.flag(R8),
        }
    }

    /// Every lever row of [`REGISTRY`], one line each: the value this
    /// reading has — as set, or the default — or `-` for a lever it does not
    /// parse or its binary does not act on, and who reads it. What `--levers`
    /// prints ([`at_main`]); a name that is no lever has no line.
    pub(crate) fn table(&self) -> String {
        let mut out = String::new();
        for row in REGISTRY.iter().filter(|r| r.is_lever()) {
            let value = match self.entries.iter().find(|e| std::ptr::eq(e.row, row)) {
                Some(Entry { acts: false, .. }) | None => "-".to_string(),
                Some(Entry {
                    set: Some(text), ..
                }) => format!("set {text}"),
                Some(Entry { set: None, .. }) => match row.default {
                    Unset::Is(d) => format!("default {d}"),
                    Unset::Means(m) => format!("unset: {m}"),
                },
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

/// What a Qwen3.8 plan runs by with [`QWEN38_EXPERTS`] unset: the card
/// experts.
pub const QWEN38_EXPERTS_UNSET: &str = "card";

/// The residency word a serving placement runs by with [`RESIDENCY`] unset.
pub const RESIDENCY_SERVING: &str = "mid-p40-s1";

/// What decides [`RESIDENCY`] unset, as the binary that reads it knows its
/// load before it opens anything ([`Levers::residency_at`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencyAt {
    /// The load's placement is one the residency machine runs over: V4.1's
    /// `--place a` or `bp`.
    pub serving_place: bool,
    /// `BLOOMERY_CHECK_FINITE=1`: the probe steps every position outside
    /// the engine's passes.
    pub check_finite: bool,
    /// `BLOOMERY_ROUTE_TRACE` is set: the trace records a fixed placement's
    /// routing.
    pub route_trace: bool,
    /// `BLOOMERY_PREFILL=steps`: each prompt id would end a decode pass the
    /// residency rule counts.
    pub prefill_steps: bool,
}

impl ResidencyAt {
    /// A load the residency machine does not run over: unset is `off`.
    pub const FIXED: ResidencyAt = ResidencyAt {
        serving_place: false,
        check_finite: false,
        route_trace: false,
        prefill_steps: false,
    };
}

/// Why a load runs the residency word it does ([`ResidencyPick`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidencyWhy {
    /// [`RESIDENCY`] is set.
    Set,
    /// Unset under a serving placement: [`RESIDENCY_SERVING`].
    Place,
    /// Unset under a placement the machine does not run over.
    FixedPlace,
    /// Unset beside the finite probe.
    CheckFinite,
    /// Unset beside the route trace.
    RouteTrace,
    /// Unset beside the step feed.
    PrefillSteps,
}

impl ResidencyWhy {
    /// The word the `residency lever` record prints.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            ResidencyWhy::Set => "set",
            ResidencyWhy::Place => "place",
            ResidencyWhy::FixedPlace => "fixed_place",
            ResidencyWhy::CheckFinite => "check_finite",
            ResidencyWhy::RouteTrace => "route_trace",
            ResidencyWhy::PrefillSteps => "prefill_steps",
        }
    }
}

/// The residency word a load runs by, and why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencyPick {
    /// `off` or `mid-p<P>-s<S>`.
    pub word: &'static str,
    /// Set, or what picked it unset.
    pub why: ResidencyWhy,
}

/// What [`RESIDENCY`] unset means at `at`: [`RESIDENCY_SERVING`] under a
/// serving placement, `off` where the machine does not run — the first of a
/// fixed placement, the finite probe, the route trace and the step feed
/// names why.
#[must_use]
pub fn residency_unset(at: ResidencyAt) -> ResidencyPick {
    let why = if !at.serving_place {
        ResidencyWhy::FixedPlace
    } else if at.check_finite {
        ResidencyWhy::CheckFinite
    } else if at.route_trace {
        ResidencyWhy::RouteTrace
    } else if at.prefill_steps {
        ResidencyWhy::PrefillSteps
    } else {
        ResidencyWhy::Place
    };
    let word = if why == ResidencyWhy::Place {
        RESIDENCY_SERVING
    } else {
        "off"
    };
    ResidencyPick { word, why }
}

/// The slots a layer the Qwen3.8 unset word frees for flips in flight.
pub const RESIDENCY38_SPARES: usize = 1;

/// What decides [`RESIDENCY`] unset in `generate_qwen3moe` before any plan:
/// the file and the run's flags ([`residency38_unset`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Residency38At {
    /// The file is a qwen4exp one.
    pub qwen38_file: bool,
    /// `--dump-taps`: a qwen3moe file's tap dump.
    pub dump_taps: bool,
    /// `--place a`: the plan on the A6000.
    pub place_a: bool,
    /// `BLOOMERY_ROUTE_TRACE` is set.
    pub route_trace: bool,
    /// `--prefill step`: each prompt id a step.
    pub prefill_step: bool,
}

/// Why a `generate_qwen3moe` load runs the residency it does with
/// [`RESIDENCY`] unset; its `Display` is the `residency unset` record's why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residency38Why {
    /// Plan (a): P half the plan's fewest card experts a layer, `fewest`.
    PlanA { fewest: usize },
    /// `--dump-taps` runs a qwen3moe file.
    DumpTaps,
    /// A qwen3moe or qwen35moe file: every routed expert on the card.
    Family,
    /// `--place gate` keeps its fixed placement.
    Gate,
    /// The route trace records a fixed placement's routing.
    RouteTrace,
    /// A step-fed prompt: each prompt id would end a pass the rule counts.
    PrefillStep,
    /// The plan holds no routed expert on the card.
    NoCardExperts,
    /// The plan leaves no routed expert on the host: the churn pool would
    /// serve none.
    NoHostExperts,
    /// The plan's fewest card experts a layer, `fewest`, leave no room for
    /// half of them pinned, the spares and one that moves.
    NoRoom { fewest: usize },
    /// The churn pool at P takes `needs` bytes of host RAM; the plan leaves
    /// `leaves`.
    HostShort { needs: u64, leaves: i128 },
    /// The churn pool at P takes `needs` bytes; the host's `MemAvailable`
    /// leaves `leaves` past the plan's own host need.
    MemShort { needs: u64, leaves: i128 },
}

impl fmt::Display for Residency38Why {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Residency38Why::PlanA { fewest } => write!(
                f,
                "unset: plan (a), P half the plan's fewest card experts a layer ({fewest})"
            ),
            Residency38Why::DumpTaps => f.write_str("unset: --dump-taps runs a qwen3moe file"),
            Residency38Why::Family => f.write_str(
                "unset: a qwen3moe or qwen35moe file holds every routed expert on the card",
            ),
            Residency38Why::Gate => f.write_str("unset: --place gate keeps its fixed placement"),
            Residency38Why::RouteTrace => {
                f.write_str("unset: BLOOMERY_ROUTE_TRACE records a fixed placement's routing")
            }
            Residency38Why::PrefillStep => f.write_str(
                "unset: --prefill step feeds each prompt id as a pass the residency rule counts",
            ),
            Residency38Why::NoCardExperts => {
                f.write_str("unset: the plan holds no routed expert on the card")
            }
            Residency38Why::NoHostExperts => f.write_str(
                "unset: the plan holds every routed expert on a card; a churn pool would serve none",
            ),
            Residency38Why::NoRoom { fewest } => write!(
                f,
                "unset: the plan's fewest card experts a layer ({fewest}) leave no room for half \
                 of them pinned, {RESIDENCY38_SPARES} spare and one that moves"
            ),
            Residency38Why::HostShort { needs, leaves } => write!(
                f,
                "unset: the churn pool needs {needs} B, the plan leaves {leaves} B"
            ),
            Residency38Why::MemShort { needs, leaves } => write!(
                f,
                "unset: the churn pool needs {needs} B, MemAvailable leaves {leaves} B past the \
                 plan's host need"
            ),
        }
    }
}

/// The residency a `generate_qwen3moe` load runs with [`RESIDENCY`] unset:
/// `pinned` seed experts a layer under `mid` with [`RESIDENCY38_SPARES`]
/// spares, `None` for `off`; and why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Residency38Pick {
    pub pinned: Option<usize>,
    pub why: Residency38Why,
}

impl Residency38Pick {
    fn off(why: Residency38Why) -> Residency38Pick {
        Residency38Pick { pinned: None, why }
    }

    /// The word the load runs by: `off` or `mid-p<P>-s<S>`.
    #[must_use]
    pub fn word(&self) -> String {
        match self.pinned {
            None => "off".to_string(),
            Some(p) => format!("mid-p{p}-s{RESIDENCY38_SPARES}"),
        }
    }
}

/// [`RESIDENCY`] unset in `generate_qwen3moe`, before the plan: `off` where
/// the machine does not run — the first of `--dump-taps`, a qwen3moe or
/// qwen35moe file, `--place gate`, the route trace and a step-fed prompt
/// names why; `None` when plan (a) decides ([`residency38_at_plan`]).
#[must_use]
pub fn residency38_unset(at: Residency38At) -> Option<Residency38Pick> {
    let why = if at.dump_taps {
        Residency38Why::DumpTaps
    } else if !at.qwen38_file {
        Residency38Why::Family
    } else if !at.place_a {
        Residency38Why::Gate
    } else if at.route_trace {
        Residency38Why::RouteTrace
    } else if at.prefill_step {
        Residency38Why::PrefillStep
    } else {
        return None;
    };
    Some(Residency38Pick::off(why))
}

/// [`RESIDENCY`] unset on plan (a), from the plan: `card_experts` its card
/// experts a layer (a layer holding none left out), `host_experts` the
/// routed experts it leaves on no card, `pool_bytes` the churn pool's bytes
/// at P pinned, `headroom` the plan's host headroom and `mem_left` what the
/// host's `MemAvailable` leaves past the plan's own host need (the load's
/// check before any upload). P is half the fewest; `off` when no layer holds
/// one, when the host holds none (the pool would serve nothing), when the
/// fewest leave no room for P pinned, the spares and one that moves, or when
/// the pool at P does not fit the headroom or `mem_left` — a default the
/// user did not set never refuses the load. An error of `pool_bytes` is the
/// call's.
pub fn residency38_at_plan<E>(
    card_experts: impl IntoIterator<Item = u64>,
    host_experts: u64,
    pool_bytes: impl FnOnce(usize) -> Result<u64, E>,
    headroom: i128,
    mem_left: i128,
) -> Result<Residency38Pick, E> {
    let Some(fewest) = card_experts.into_iter().filter(|&n| n > 0).min() else {
        return Ok(Residency38Pick::off(Residency38Why::NoCardExperts));
    };
    if host_experts == 0 {
        return Ok(Residency38Pick::off(Residency38Why::NoHostExperts));
    }
    let fewest = usize::try_from(fewest).unwrap_or(usize::MAX);
    let pinned = fewest / 2;
    if pinned == 0 || fewest < pinned + RESIDENCY38_SPARES + 1 {
        return Ok(Residency38Pick::off(Residency38Why::NoRoom { fewest }));
    }
    let needs = pool_bytes(pinned)?;
    if headroom < i128::from(needs) {
        return Ok(Residency38Pick::off(Residency38Why::HostShort {
            needs,
            leaves: headroom,
        }));
    }
    if mem_left < i128::from(needs) {
        return Ok(Residency38Pick::off(Residency38Why::MemShort {
            needs,
            leaves: mem_left,
        }));
    }
    Ok(Residency38Pick {
        pinned: Some(pinned),
        why: Residency38Why::PlanA { fewest },
    })
}

/// What decides [`DRAFT`] unset on a qwen4exp file in `generate_qwen3moe`
/// ([`draft38_unset`]).
#[derive(Clone, Copy, Debug)]
pub struct Draft38At<'a> {
    /// `--place a`: the plan on the A6000.
    pub place_a: bool,
    /// `--logits`: the step head's row, which a drafted run's last call
    /// does not leave.
    pub logits: bool,
    /// `BLOOMERY_ROUTE_TRACE` is set.
    pub route_trace: bool,
    /// The MTP draft file the run would open, and whether a regular file is
    /// there.
    pub file: &'a Path,
    pub file_is_there: bool,
    /// The positions the drafted run's last window needs at most over the
    /// run's arms, and `--ctx`.
    pub need: usize,
    pub ctx: usize,
}

/// Why a qwen4exp run of `generate_qwen3moe` drafts nothing; its `Display`
/// is the `load draft=off` record's why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Draft38Off {
    /// `BLOOMERY_DRAFT=off`.
    Set,
    /// Unset under `--place gate`.
    Gate,
    /// Unset beside `--logits`.
    Logits,
    /// Unset beside the route trace.
    RouteTrace,
    /// Unset, and no regular file where the draft would be opened.
    NoFile(PathBuf),
    /// Unset, and the last window needs `need` positions of `ctx`.
    Ctx { need: usize, ctx: usize },
    /// Unset on the Qwen3.8 seat, and the target's `name`, a matrix the
    /// draft borrows, is `ty` (or `absent`), not the Q8_0 the draft reads.
    Borrowed { name: String, ty: String },
}

impl fmt::Display for Draft38Off {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Draft38Off::Set => f.write_str("BLOOMERY_DRAFT=off"),
            Draft38Off::Gate => f.write_str("unset: --place gate"),
            Draft38Off::Logits => f.write_str(
                "unset: --logits reads the step head's row; a drafted run ends on a verify",
            ),
            Draft38Off::RouteTrace => {
                f.write_str("unset: BLOOMERY_ROUTE_TRACE records one step a position")
            }
            Draft38Off::NoFile(p) => write!(f, "no file at {}", p.display()),
            Draft38Off::Ctx { need, ctx } => write!(
                f,
                "unset: the last window needs {need} positions, past --ctx {ctx}"
            ),
            Draft38Off::Borrowed { name, ty } => write!(
                f,
                "unset: the target's {name} is {ty}; the MTP draft reads it as Q8_0"
            ),
        }
    }
}

/// [`DRAFT`] unset on a qwen4exp file in `generate_qwen3moe`: the MTP draft
/// (`None`) under `--place a` when its file is there; else why not — the
/// first of `--place gate`, `--logits`, the route trace, no file and a last
/// window past `--ctx`. None of them refuses the run: the plain path runs.
#[must_use]
pub fn draft38_unset(at: &Draft38At<'_>) -> Option<Draft38Off> {
    if !at.place_a {
        Some(Draft38Off::Gate)
    } else if at.logits {
        Some(Draft38Off::Logits)
    } else if at.route_trace {
        Some(Draft38Off::RouteTrace)
    } else if !at.file_is_there {
        Some(Draft38Off::NoFile(at.file.to_path_buf()))
    } else if at.need > at.ctx {
        Some(Draft38Off::Ctx {
            need: at.need,
            ctx: at.ctx,
        })
    } else {
        None
    }
}

/// The word the GLM seat of `bloomery-serve` drafts by with [`DRAFT`]
/// unset where the draft can run: `mtp`, the file's NextN layer
/// ([`glm_unset`]).
pub const GLM_DRAFT_UNSET: &str = "mtp";

/// The word the GLM seat runs by with [`RESIDENCY`] unset where the machine
/// runs: the serving placements', no seed expert pinned and one spare a layer
/// ([`glm_unset`], [`glm_residency_at_plan`]).
pub const GLM_RESIDENCY_UNSET: &str = "mid-p0-s1";

/// What decides the GLM seat's unset words before any plan ([`glm_unset`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlmAt {
    /// The serving placements: the A6000 is the stage card, alone (`--place
    /// a`) or with expert tier cards beside it (`bp`); `gate` is not one.
    pub serving_place: bool,
    /// The file's next-token layers (`block_count` less the trunk's); the
    /// NextN draft runs one.
    pub nextn_layers: usize,
    /// The positions one drafted window needs from an empty model at most,
    /// and the stores' positions (`--ctx`).
    pub need: usize,
    pub ctx: usize,
    /// `--prefill steps`: each prompt id would end a pass the residency rule
    /// counts.
    pub prefill_steps: bool,
}

/// Why the GLM seat runs the unset word it does; its `Display` is the
/// `draft unset`, `load draft=off` and `residency unset` records' why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlmWhy {
    /// A serving placement (`--place a` or `bp`) runs the default word.
    Serving,
    /// `--place gate` keeps its fixed placement and drafts nothing.
    Gate,
    /// The file carries `layers` next-token layers, not the one the draft
    /// runs.
    Nextn { layers: usize },
    /// One window needs `need` positions of `ctx`: the stores are short of
    /// a window.
    Ctx { need: usize, ctx: usize },
    /// The prompt is fed by steps.
    PrefillSteps,
    /// The plan holds no routed expert on the card.
    NoCardExperts,
    /// The plan's fewest card experts a layer, `fewest`, leave no room for
    /// the word's pinned experts, its spares and one that moves.
    NoRoom { fewest: u64 },
    /// The churn pool takes `needs` bytes of host RAM; the plan leaves
    /// `leaves`.
    HostShort { needs: u64, leaves: i128 },
    /// The churn pool takes `needs` bytes; the host's `MemAvailable` leaves
    /// `leaves` past the load's own host need.
    MemShort { needs: u64, leaves: i128 },
}

impl fmt::Display for GlmWhy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            GlmWhy::Serving => f.write_str("unset: --place a or bp"),
            GlmWhy::Gate => f.write_str("unset: --place gate keeps its fixed placement"),
            GlmWhy::Nextn { layers } => write!(
                f,
                "unset: the file carries {layers} next-token layers; the NextN draft runs one"
            ),
            GlmWhy::Ctx { need, ctx } => write!(
                f,
                "unset: one window needs {need} positions, past --ctx {ctx}"
            ),
            GlmWhy::PrefillSteps => f.write_str(
                "unset: --prefill steps feeds each prompt id as a pass the residency rule counts",
            ),
            GlmWhy::NoCardExperts => {
                f.write_str("unset: the plan holds no routed expert on the card")
            }
            GlmWhy::NoRoom { fewest } => write!(
                f,
                "unset: the plan's fewest card experts a layer ({fewest}) leave no room for the \
                 word's pinned experts, its spares and one that moves"
            ),
            GlmWhy::HostShort { needs, leaves } => write!(
                f,
                "unset: the churn pool needs {needs} B, the plan leaves {leaves} B"
            ),
            GlmWhy::MemShort { needs, leaves } => write!(
                f,
                "unset: the churn pool needs {needs} B, MemAvailable leaves {leaves} B past the \
                 load's host need"
            ),
        }
    }
}

/// One unset lever of the GLM seat: the word it runs by, and why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlmPick {
    pub word: &'static str,
    pub why: GlmWhy,
}

impl GlmPick {
    fn off(why: GlmWhy) -> GlmPick {
        GlmPick { word: "off", why }
    }
}

/// The GLM seat's [`DRAFT`] and [`RESIDENCY`] unset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlmUnset {
    pub draft: GlmPick,
    pub residency: GlmPick,
}

/// [`DRAFT`] and [`RESIDENCY`] unset on the GLM seat at `at`, before the
/// plan: under a serving placement (`--place a` or `bp`)
/// [`GLM_DRAFT_UNSET`] when the file carries the one
/// next-token layer and [`GLM_RESIDENCY_UNSET`], which the plan then decides
/// ([`glm_residency_at_plan`]); `off`, with why, under `--place gate` and
/// with stores too short for one window (both), for the draft on a file of
/// other than one next-token layer, and for the residency beside the step
/// feed. None of them refuses the load.
#[must_use]
pub fn glm_unset(at: GlmAt) -> GlmUnset {
    let short = at.need > at.ctx;
    let draft = if !at.serving_place {
        GlmPick::off(GlmWhy::Gate)
    } else if at.nextn_layers != 1 {
        GlmPick::off(GlmWhy::Nextn {
            layers: at.nextn_layers,
        })
    } else if short {
        GlmPick::off(GlmWhy::Ctx {
            need: at.need,
            ctx: at.ctx,
        })
    } else {
        GlmPick {
            word: GLM_DRAFT_UNSET,
            why: GlmWhy::Serving,
        }
    };
    let residency = if !at.serving_place {
        GlmPick::off(GlmWhy::Gate)
    } else if short {
        GlmPick::off(GlmWhy::Ctx {
            need: at.need,
            ctx: at.ctx,
        })
    } else if at.prefill_steps {
        GlmPick::off(GlmWhy::PrefillSteps)
    } else {
        GlmPick {
            word: GLM_RESIDENCY_UNSET,
            why: GlmWhy::Serving,
        }
    };
    GlmUnset { draft, residency }
}

/// [`glm_unset`]'s residency on the plan: `pick` as it is unless it runs the
/// machine; then `off`, with why, when the plan holds no card expert
/// (`card_experts` a layer, a layer holding none left out), when its fewest
/// leave no room for the word's pinned experts, its spares and one that
/// moves, or when the churn pool at the word's pinned count (`pool_bytes`)
/// does not fit `headroom` (the plan's host headroom less what the load
/// hosts beside it) or `mem_left` (what `MemAvailable` leaves past the
/// load's host need) — a default the user did not set never refuses the
/// load. An error of `pool_bytes` is the call's.
pub fn glm_residency_at_plan<E>(
    pick: GlmPick,
    card_experts: impl IntoIterator<Item = u64>,
    pool_bytes: impl FnOnce(usize) -> Result<u64, E>,
    headroom: i128,
    mem_left: i128,
) -> Result<GlmPick, E> {
    let Some(ResidencyWord::Mid { pinned, spares }) = residency_word(pick.word) else {
        return Ok(pick);
    };
    let Some(fewest) = card_experts.into_iter().filter(|&n| n > 0).min() else {
        return Ok(GlmPick::off(GlmWhy::NoCardExperts));
    };
    if usize::try_from(fewest).unwrap_or(usize::MAX) < pinned + spares + 1 {
        return Ok(GlmPick::off(GlmWhy::NoRoom { fewest }));
    }
    let needs = pool_bytes(pinned)?;
    if headroom < i128::from(needs) {
        return Ok(GlmPick::off(GlmWhy::HostShort {
            needs,
            leaves: headroom,
        }));
    }
    if mem_left < i128::from(needs) {
        return Ok(GlmPick::off(GlmWhy::MemShort {
            needs,
            leaves: mem_left,
        }));
    }
    Ok(pick)
}

/// The host tier's load settings, from a binary's one reading
/// ([`Levers::host`]): what a placed load does to its plan's host set and to
/// the card segments' file pages, and where the host tier reads a V4.1 file's
/// routed gates and ups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostCfg {
    /// [`HOST_POPULATE`]: the plan's host set is read into the page cache
    /// and mapped at load; `false` leaves it to the steps' first touches.
    pub populate: bool,
    /// [`HOST_LOCK`]: the host set is locked in RAM for the model's
    /// lifetime.
    pub lock: bool,
    /// [`CARD_DONTNEED`]: each card segment's file pages leave the page cache
    /// once uploaded.
    pub card_dontneed: bool,
    /// [`R8`]: a V4.1 host tier reads its routed gates and ups from the r8
    /// sidecar when there is one; `false` reads the source's, the
    /// same-binary arm.
    pub r8: bool,
}

/// A binary's one reading of the levers, first thing in `main`. `acts_on`
/// names the Parsed levers its run acts on — the pool's two, [`THREADS`] and
/// [`SPIN`], it always does — and a binary whose child inherits its
/// environment names the child's too. Refused, with every refusal at once: a
/// Parsed lever set to a value its kind does not take, or set when `acts_on`
/// leaves it out; a retired name that is set; a `BLOOMERY_*` name no row
/// names. A name in `acts_on` that is not a Parsed lever panics by name — a
/// caller's mistake. With `--levers` among the process's arguments it prints
/// the reading's table — `-` for a lever the binary does not act on — on
/// stdout and exits 0 instead.
pub fn at_main(acts_on: &[&str]) -> Result<Levers, LeverError> {
    let mut args = std::env::args_os();
    let bin = args
        .next()
        .as_deref()
        .map(Path::new)
        .and_then(Path::file_name)
        .map_or_else(|| "unnamed".to_string(), lossy);
    let env: Vec<(OsString, OsString)> = std::env::vars_os().collect();
    let levers = read(&env, Scope::Main { bin: &bin, acts_on })?;
    if args.any(|a| a == "--levers") {
        print!("{}", levers.table());
        std::process::exit(0);
    }
    Ok(levers)
}

/// The worker pool's two levers ([`THREADS`], [`SPIN`]), read where the
/// process-wide pool is built ([`pool_levers`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolLevers {
    /// The pool's thread count; `None` unset (the physical core count).
    pub threads: Option<usize>,
    /// Spin iterations a waiting pool thread makes before it parks.
    pub spin: u64,
}

/// The worker pool's levers from the process environment: [`THREADS`] and
/// [`SPIN`] parsed as [`at_main`] parses them, and refused, with every
/// refusal at once, when one is set to a value its kind does not take and
/// when a retired name is set. Nothing else: which levers a binary acts on,
/// and which names it knows, is its `main`'s reading.
pub fn pool_levers() -> Result<PoolLevers, LeverError> {
    let env: Vec<(OsString, OsString)> = std::env::vars_os().collect();
    let levers = read(&env, Scope::Pool)?;
    Ok(PoolLevers {
        threads: levers.threads(),
        spin: levers.spin(),
    })
}

/// The registry as a Markdown table — name, class, what it takes, what unset
/// means, who reads it, what it does: the repository's documentation of every
/// `BLOOMERY_*` name, the levers first.
#[must_use]
pub fn markdown() -> String {
    let mut out = String::from(
        "| name | class | takes | unset | read by | what it does |\n|---|---|---|---|---|---|\n",
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

/// Why [`parse_bytes`] refused a value, with the value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BytesError {
    /// Not digits with optional `_` separators, then nothing, `M` or `G`.
    NotBytes(String),
    /// Bytes past `u64`.
    Overflow(String),
}

impl BytesError {
    /// Why, without the value.
    fn reason(&self) -> &'static str {
        match self {
            BytesError::NotBytes(_) => {
                "not bytes, or a whole number of MiB or GiB with an M or G suffix"
            }
            BytesError::Overflow(_) => OVERFLOW,
        }
    }
}

impl fmt::Display for BytesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (BytesError::NotBytes(v) | BytesError::Overflow(v)) = self;
        write!(f, "{v:?} is {}", self.reason())
    }
}

impl std::error::Error for BytesError {}

/// Bytes `value` names: digits with optional `_` separators, then nothing
/// (bytes), `M` (MiB) or `G` (GiB). Anything else is [`BytesError::NotBytes`],
/// and a value past `u64` [`BytesError::Overflow`].
pub fn parse_bytes(value: &str) -> Result<u64, BytesError> {
    let (digits, unit) = match value.strip_suffix('G') {
        Some(d) => (d, 1u64 << 30),
        None => match value.strip_suffix('M') {
            Some(d) => (d, 1u64 << 20),
            None => (value, 1),
        },
    };
    let digits: String = digits.chars().filter(|&c| c != '_').collect();
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(BytesError::NotBytes(value.to_string()));
    }
    digits
        .parse::<u64>()
        .ok()
        .and_then(|n| n.checked_mul(unit))
        .ok_or_else(|| BytesError::Overflow(value.to_string()))
}
