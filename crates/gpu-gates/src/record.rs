//! The record lines of the V4.1 binaries — `generate_ds41`, `bloomery-chat`,
//! `bloomery-serve-ds41` and the prefill gate's stat line —, of
//! `generate_glm5next` and of `generate_qwen3moe`'s tap dump and stat lines,
//! and the one owner of each line's syntax: its [`Kind`].
//!
//! A kind is the words a line opens with (its head) and its parts in order:
//! ` name=value`, a bare value, a flag word, or literal text between values.
//! [`Record`] renders one line through its kind, part by part, and panics,
//! naming the kind and the part, on a value given out of order, left out, or
//! of another type than the kind declares: a line whose syntax is not its
//! kind's is never printed. Each binary answers `--records-schema` right after
//! its levers ([`at_main`]) with every kind it prints, one JSON object a line;
//! `tools/bloomery/records.py` parses a log by that schema, so a reader names a
//! kind and its fields, never a column or a pattern. The same call registers
//! those kinds, and a line of any other kind is refused by name: the schema
//! is every line the binary can print. [`Log`] reads a log by the same rules
//! in Rust, a record's values by field name ([`Fields`]), and refuses by name
//! ([`ReadError`]) a record absent where one is due, a line that opens a
//! kind's record and does not read whole, and a field the kind lacks, of
//! another type, or left out.
//!
//! The text is the one the lines' readers know: `<head> name=value …`, the
//! head a fixed run of words and the pairs in a fixed order, so a reader that
//! greps a head or cuts a `name=` keeps working. A value's unit is the
//! schema's `unit`; where the text puts a unit after a bare value (`in 48.2
//! s`, `(1342177280 B)`), the schema's name for that value carries it
//! (`load_s`, `card_expert_bytes`).

use std::fmt::{Debug, Display, Write as _};
use std::sync::OnceLock;

#[cfg(feature = "gpu")]
use bloomery_gpu::host::PassKind;
#[cfg(feature = "gpu")]
use bloomery_gpu::host::swap::{CallPick, CallReport, Leak, PassReport, ResetReport};
#[cfg(feature = "gpu")]
use bloomery_gpu::host::xstream::{XLayer, XReport};
#[cfg(feature = "gpu")]
use bloomery_gpu::hybrid::HostResidency;
#[cfg(feature = "gpu")]
use bloomery_gpu::prompt_timing::PromptStats;
use model::placement::{Machine, Plan};

/// The schema's version, the first line of `--records-schema`.
pub const VERSION: u32 = 1;

/// What a value part holds, and how it is written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    /// An unsigned integer.
    U64,
    /// A signed integer.
    I64,
    /// A float with this many decimals.
    F64(usize),
    /// `true` or `false`.
    Bool,
    /// One word: no whitespace.
    Word,
    /// Text that may hold spaces, up to the next part (or the line's end).
    Text,
    /// A list as Rust debug-prints a slice of integers: `[a, b, c]`.
    List,
    /// A list with no spaces: `[a,b,c]`.
    Csv,
}

impl Ty {
    fn name(self) -> String {
        match self {
            Ty::U64 => "u64".into(),
            Ty::I64 => "i64".into(),
            Ty::F64(d) => format!("f64.{d}"),
            Ty::Bool => "bool".into(),
            Ty::Word => "word".into(),
            Ty::Text => "text".into(),
            Ty::List => "list".into(),
            Ty::Csv => "csv".into(),
        }
    }
}

/// A value of a line: its name in the schema (and in the text, for a
/// [`Part::Key`]), its type, its unit (empty for a count or a label), and
/// whether the line may leave it out.
#[derive(Clone, Copy, Debug)]
pub struct Field {
    pub name: &'static str,
    pub ty: Ty,
    pub unit: &'static str,
    pub opt: bool,
}

/// One part of a line, in order after its head. An optional part carries its
/// own leading space and no literal belongs to it, so leaving it out leaves
/// the rest of the line as it is.
#[derive(Clone, Copy, Debug)]
pub enum Part {
    /// ` name=value`.
    Key(Field),
    /// The value alone; the literals around it hold its spacing.
    Pos(Field),
    /// ` name` when set, nothing when not.
    Flag(&'static str),
    /// Text between values.
    Lit(&'static str),
}

impl Part {
    fn value_name(&self) -> Option<&'static str> {
        match self {
            Part::Key(f) | Part::Pos(f) => Some(f.name),
            Part::Flag(n) => Some(*n),
            Part::Lit(_) => None,
        }
    }

    fn optional(&self) -> bool {
        match self {
            Part::Key(f) | Part::Pos(f) => f.opt,
            Part::Flag(_) => true,
            Part::Lit(_) => false,
        }
    }
}

const fn key(name: &'static str, ty: Ty, unit: &'static str) -> Part {
    Part::Key(Field {
        name,
        ty,
        unit,
        opt: false,
    })
}

const fn opt(name: &'static str, ty: Ty, unit: &'static str) -> Part {
    Part::Key(Field {
        name,
        ty,
        unit,
        opt: true,
    })
}

const fn pos(name: &'static str, ty: Ty, unit: &'static str) -> Part {
    Part::Pos(Field {
        name,
        ty,
        unit,
        opt: false,
    })
}

const fn lit(text: &'static str) -> Part {
    Part::Lit(text)
}

const fn flag(name: &'static str) -> Part {
    Part::Flag(name)
}

/// A kind of line: the name the schema and `records.py` call it by, the words
/// it opens with, its parts, and one sentence on what it says.
#[derive(Debug)]
pub struct Kind {
    pub name: &'static str,
    pub head: &'static str,
    pub parts: &'static [Part],
    pub doc: &'static str,
}

impl Kind {
    /// The kind as one JSON object: `records.py`'s input.
    fn json(&self) -> String {
        let mut parts = Vec::with_capacity(self.parts.len());
        for p in self.parts {
            parts.push(match p {
                Part::Key(f) | Part::Pos(f) => format!(
                    "{{\"{}\":{},\"ty\":\"{}\",\"unit\":{}{}}}",
                    if matches!(p, Part::Key(_)) {
                        "key"
                    } else {
                        "pos"
                    },
                    json_str(f.name),
                    f.ty.name(),
                    json_str(f.unit),
                    if f.opt { ",\"opt\":true" } else { "" }
                ),
                Part::Flag(n) => format!("{{\"flag\":{}}}", json_str(n)),
                Part::Lit(t) => format!("{{\"lit\":{}}}", json_str(t)),
            });
        }
        format!(
            "{{\"kind\":{},\"head\":{},\"doc\":{},\"parts\":[{}]}}",
            json_str(self.name),
            json_str(self.head),
            json_str(self.doc),
            parts.join(",")
        )
    }
}

/// `s` as a JSON string.
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A binary and the kinds it prints, as its [`at_main`] registered them.
#[derive(Clone, Copy, Debug)]
struct Prints {
    bin: &'static str,
    kinds: &'static [&'static Kind],
}

impl Prints {
    fn holds(&self, kind: &Kind) -> bool {
        self.kinds.iter().any(|k| std::ptr::eq(*k, kind))
    }
}

/// The process's [`Prints`], set once by [`at_main`]; unset, a line of any
/// kind renders.
static PRINTS: OnceLock<Prints> = OnceLock::new();

/// With `--records-schema` among the process's arguments, print `bin`'s
/// kinds ([`print_schema`]) and exit 0; otherwise register them, after which
/// [`Record::line`] refuses a line of any other kind. Called right after
/// `bloomery_levers::at_main`, before any argument parse or load; a second
/// call with another binary or list panics by name.
pub fn at_main(bin: &'static str, kinds: &'static [&'static Kind]) {
    if std::env::args_os().skip(1).any(|a| a == "--records-schema") {
        print_schema(bin, kinds);
        std::process::exit(0);
    }
    let held = PRINTS.get_or_init(|| Prints { bin, kinds });
    assert!(
        held.bin == bin && std::ptr::eq(held.kinds, kinds),
        "record::at_main({bin}): the process registered {}'s kinds already",
        held.bin
    );
    #[cfg(feature = "gpu")]
    if kinds.iter().any(|k| std::ptr::eq(*k, &RESIDENCY_LEAK)) {
        bloomery_gpu::host::swap::set_leak_sink(eprint_leak);
    }
}

/// The residency machine's leak sink: its `residency leak` line on stderr.
#[cfg(feature = "gpu")]
fn eprint_leak(l: &Leak) {
    residency_leak(l).eprint();
}

/// The schema of `bin`'s lines on stdout ([`schema`]).
pub fn print_schema(bin: &str, kinds: &[&Kind]) {
    print!("{}", schema(bin, kinds));
}

/// The schema of `bin`'s lines: a header object, then one object a kind, in
/// the order the binary prints them, a line each. `tools/bloomery/schema/`
/// holds each binary's, which `tools/bloomery/records.py` reads.
#[must_use]
pub fn schema(bin: &str, kinds: &[&Kind]) -> String {
    let mut out = format!(
        "{{\"records\":{VERSION},\"bin\":{},\"kinds\":{}}}\n",
        json_str(bin),
        kinds.len()
    );
    for k in kinds {
        out.push_str(&k.json());
        out.push('\n');
    }
    out
}

/// One line of a [`Kind`], rendered part by part: each value named in the
/// kind's order, the literals between them written as they come, an optional
/// part left out by naming a later one.
#[must_use]
pub struct Record {
    kind: &'static Kind,
    at: usize,
    line: String,
}

impl Record {
    pub fn new(kind: &'static Kind) -> Record {
        Record {
            kind,
            at: 0,
            line: kind.head.to_string(),
        }
    }

    fn refuse(&self, what: &str) -> ! {
        panic!(
            "record {}: {what} (line so far: {:?})",
            self.kind.name, self.line
        )
    }

    /// The part `name` next, with the literals before it written and the
    /// optional parts before it left out.
    fn next(&mut self, name: &str) -> Part {
        while let Some(p) = self.kind.parts.get(self.at) {
            self.at += 1;
            match p {
                Part::Lit(t) => self.line.push_str(t),
                _ if p.value_name() == Some(name) => return *p,
                _ if p.optional() => {}
                _ => self.refuse(&format!(
                    "{name} given where {} is due",
                    p.value_name().unwrap_or("?")
                )),
            }
        }
        self.refuse(&format!("no part {name} left"))
    }

    /// The value part `name` next, and its field.
    fn field(&mut self, name: &str) -> (Part, Field) {
        match self.next(name) {
            p @ (Part::Key(f) | Part::Pos(f)) => (p, f),
            Part::Flag(_) | Part::Lit(_) => self.refuse(&format!("{name} is not a value part")),
        }
    }

    /// `text` as the value of `p`, refused unless `fits` its type.
    fn write(&mut self, p: Part, f: Field, text: &str, fits: bool) {
        if !fits {
            self.refuse(&format!("{} = {text:?} is not a {}", f.name, f.ty.name()));
        }
        if let Part::Key(_) = p {
            self.line.push(' ');
            self.line.push_str(f.name);
            self.line.push('=');
        }
        self.line.push_str(text);
    }

    /// An integer part.
    pub fn u(mut self, name: &str, v: impl Display) -> Record {
        let text = v.to_string();
        let (p, f) = self.field(name);
        let neg = text.starts_with('-');
        let digits = text.strip_prefix('-').unwrap_or(&text);
        let int = !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit());
        let fits = int && (f.ty == Ty::I64 || (f.ty == Ty::U64 && !neg));
        self.write(p, f, &text, fits);
        self
    }

    /// A float part, with the kind's decimals.
    pub fn f(mut self, name: &str, v: f64) -> Record {
        let (p, f) = self.field(name);
        let Ty::F64(d) = f.ty else {
            self.refuse(&format!("{name} is not a float part"));
        };
        self.write(p, f, &format!("{v:.d$}"), true);
        self
    }

    /// A word, text or bool part, as `v` displays.
    pub fn w(mut self, name: &str, v: impl Display) -> Record {
        let text = v.to_string();
        let (p, f) = self.field(name);
        let fits = match f.ty {
            Ty::Word => !text.is_empty() && !text.contains(char::is_whitespace),
            Ty::Text => true,
            Ty::Bool => text == "true" || text == "false",
            _ => false,
        };
        self.write(p, f, &text, fits);
        self
    }

    /// A `[a, b, c]` part.
    pub fn list<T: Debug>(mut self, name: &str, v: &[T]) -> Record {
        let (p, f) = self.field(name);
        self.write(p, f, &format!("{v:?}"), f.ty == Ty::List);
        self
    }

    /// A `[a,b,c]` part.
    pub fn csv<T: Display>(mut self, name: &str, v: impl IntoIterator<Item = T>) -> Record {
        let items: Vec<String> = v.into_iter().map(|x| x.to_string()).collect();
        let (p, f) = self.field(name);
        self.write(p, f, &format!("[{}]", items.join(",")), f.ty == Ty::Csv);
        self
    }

    /// A flag: ` name` when `on`.
    pub fn flag(mut self, name: &str, on: bool) -> Record {
        match self.next(name) {
            Part::Flag(n) => {
                if on {
                    self.line.push(' ');
                    self.line.push_str(n);
                }
                self
            }
            _ => self.refuse(&format!("{name} is not a flag")),
        }
    }

    /// The line, with the literals after the last value written; refused
    /// while a part the kind requires is missing, and when the process
    /// registered its binary's kinds ([`at_main`]) and this one is not among
    /// them.
    #[must_use]
    pub fn line(self) -> String {
        self.line_under(PRINTS.get())
    }

    /// [`Record::line`] under the registration `prints`.
    fn line_under(mut self, prints: Option<&Prints>) -> String {
        if let Some(held) = prints
            && !held.holds(self.kind)
        {
            self.refuse(&format!(
                "{} does not print this kind: it is not among the kinds its \
                 record::at_main registered",
                held.bin
            ));
        }
        while let Some(p) = self.kind.parts.get(self.at) {
            self.at += 1;
            match p {
                Part::Lit(t) => self.line.push_str(t),
                _ if p.optional() => {}
                _ => self.refuse(&format!("{} left out", p.value_name().unwrap_or("?"))),
            }
        }
        self.line
    }

    /// The line on stdout.
    pub fn print(self) {
        println!("{}", self.line());
    }

    /// The line on stderr.
    pub fn eprint(self) {
        eprintln!("{}", self.line());
    }
}

// ---------------------------------------------------------------- the reader

/// A read of a [`Log`] or of a [`Fields`] value refused, by name: what a
/// line-splitting reader would read as nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReadError {
    /// The kind is not among the kinds the log is read by.
    NotPrinted { log: String, kind: &'static str },
    /// The log holds no record of the kind.
    Absent { log: String, kind: &'static str },
    /// The log holds `n` records of a kind read as one.
    Many {
        log: String,
        kind: &'static str,
        n: usize,
    },
    /// Line `at` opens a record of the kind (its head, then its first field,
    /// or any of its fields when no other kind has the head) and no kind
    /// reads it whole.
    Broken {
        log: String,
        kind: &'static str,
        at: usize,
        line: String,
    },
    /// The kind has no value part of that name.
    NoField { kind: &'static str, field: String },
    /// The record leaves out the optional part read as present.
    Missing {
        kind: &'static str,
        field: &'static str,
        line: String,
    },
    /// The part is of another type than the read asks for.
    Type {
        kind: &'static str,
        field: &'static str,
        ty: Ty,
        asked: &'static str,
    },
    /// The value is past its type's range.
    Value {
        kind: &'static str,
        field: &'static str,
        value: String,
    },
}

impl Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadError::NotPrinted { log, kind } => {
                write!(f, "{log}: `{kind}` is not among the kinds it is read by")
            }
            ReadError::Absent { log, kind } => write!(f, "{log}: no `{kind}` record"),
            ReadError::Many { log, kind, n } => {
                write!(f, "{log}: {n} `{kind}` records where one is due")
            }
            ReadError::Broken {
                log,
                kind,
                at,
                line,
            } => write!(
                f,
                "{log}: line {at} opens a `{kind}` record and does not read whole: {line}"
            ),
            ReadError::NoField { kind, field } => write!(f, "`{kind}` has no field {field}"),
            ReadError::Missing { kind, field, line } => {
                write!(f, "the `{kind}` record leaves out {field}: {line}")
            }
            ReadError::Type {
                kind,
                field,
                ty,
                asked,
            } => write!(f, "`{kind}`'s {field} is a {}, read as {asked}", ty.name()),
            ReadError::Value { kind, field, value } => {
                write!(f, "the `{kind}` record's {field}={value} is out of range")
            }
        }
    }
}

impl std::error::Error for ReadError {}

/// A log read by the kinds of the binary that wrote it, line by line as
/// `tools/bloomery/records.py` reads one: the kinds whose head opens the line
/// (as a word), the longest head first, and of them the first whose whole
/// pattern reads the line — so kinds that share a head (`plan` and `plan38`,
/// `cache` and `cache save`) are told apart by the whole line. A line that
/// no kind reads whole but that opens a kind's record (records.py's loose
/// read: its head, then its first field, or any of its fields when no other
/// kind has the head) is broken: every read of that kind is refused by name.
/// Any other line is no record.
#[derive(Debug)]
pub struct Log {
    what: String,
    kinds: Vec<&'static Kind>,
    entries: Vec<Entry>,
}

#[derive(Debug)]
enum Entry {
    Whole(Fields),
    /// Line `at` (from 1), and every kind whose record it opens.
    Broken {
        at: usize,
        line: String,
        kinds: Vec<&'static Kind>,
    },
}

impl Log {
    /// The records of `text` by `kinds`, the list the binary that wrote it
    /// registered; the log is `the log` in an error until [`Log::named`].
    #[must_use]
    pub fn of(text: &str, kinds: &[&'static Kind]) -> Log {
        let mut order = kinds.to_vec();
        order.sort_by_key(|k| std::cmp::Reverse(k.head.len()));
        let entries = text
            .lines()
            .enumerate()
            .filter_map(|(i, line)| read_line(&order, i + 1, line))
            .collect();
        Log {
            what: "the log".to_owned(),
            kinds: kinds.to_vec(),
            entries,
        }
    }

    /// The log as its errors name it.
    #[must_use]
    pub fn named(mut self, what: impl Into<String>) -> Log {
        self.what = what.into();
        self
    }

    /// Every record of `kind`, in order; refused when `kind` is not among the
    /// log's kinds or a line opens one and does not read whole.
    pub fn all(&self, kind: &Kind) -> Result<Vec<Fields>, ReadError> {
        if !self.kinds.iter().any(|k| std::ptr::eq(*k, kind)) {
            return Err(ReadError::NotPrinted {
                log: self.what.clone(),
                kind: kind.name,
            });
        }
        let mut out = Vec::new();
        for e in &self.entries {
            match e {
                Entry::Whole(f) if std::ptr::eq(f.kind, kind) => out.push(f.clone()),
                Entry::Broken { at, line, kinds }
                    if kinds.iter().any(|k| std::ptr::eq(*k, kind)) =>
                {
                    return Err(ReadError::Broken {
                        log: self.what.clone(),
                        kind: kind.name,
                        at: *at,
                        line: line.clone(),
                    });
                }
                _ => {}
            }
        }
        Ok(out)
    }

    /// The first record of `kind`, `None` when there is none; refused as
    /// [`Log::all`] is.
    pub fn first(&self, kind: &Kind) -> Result<Option<Fields>, ReadError> {
        Ok(self.all(kind)?.into_iter().next())
    }

    /// The one record of `kind`; none, or more than one, is refused by name.
    pub fn one(&self, kind: &Kind) -> Result<Fields, ReadError> {
        let mut all = self.all(kind)?;
        match all.len() {
            1 => Ok(all.remove(0)),
            0 => Err(ReadError::Absent {
                log: self.what.clone(),
                kind: kind.name,
            }),
            n => Err(ReadError::Many {
                log: self.what.clone(),
                kind: kind.name,
                n,
            }),
        }
    }
}

/// `line` (number `at`) by the kinds in `order`, the longest head first.
fn read_line(order: &[&'static Kind], at: usize, line: &str) -> Option<Entry> {
    let opened: Vec<&'static Kind> = order.iter().copied().filter(|k| k.opens(line)).collect();
    if let Some(f) = opened.iter().find_map(|k| k.read(at, line)) {
        return Some(Entry::Whole(f));
    }
    let kinds: Vec<&'static Kind> = opened
        .iter()
        .copied()
        .filter(|k| {
            k.opens_fields(
                line,
                opened.iter().filter(|o| o.head == k.head).count() == 1,
            )
        })
        .collect();
    (!kinds.is_empty()).then(|| Entry::Broken {
        at,
        line: line.to_owned(),
        kinds,
    })
}

impl Kind {
    /// Whether `line` opens with the head as a word: a space or the line's
    /// end after it, unless the head ends in `=` or `:`.
    fn opens(&self, line: &str) -> bool {
        line.strip_prefix(self.head).is_some_and(|rest| {
            rest.is_empty() || rest.starts_with(' ') || self.head.ends_with(['=', ':'])
        })
    }

    /// `line` (number `at`) read whole by the parts.
    fn read(&'static self, at: usize, line: &str) -> Option<Fields> {
        let mut values = Vec::new();
        walk(self.parts, &line[self.head.len()..], &mut values).then(|| Fields {
            kind: self,
            at,
            line: line.to_owned(),
            values,
        })
    }

    /// Whether `line`, which opens with the head, goes on with ` name=` of
    /// the kind's first field — or, `alone` (no other kind has the head), of
    /// any of its key fields.
    fn opens_fields(&self, line: &str, alone: bool) -> bool {
        let first = self.parts.iter().find_map(|p| match p {
            Part::Lit(_) => None,
            Part::Key(f) => Some(Some(f.name)),
            Part::Pos(_) | Part::Flag(_) => Some(None),
        });
        let (Some(Some(first)), Some(name)) = (first, key_at(&line[self.head.len()..])) else {
            return false;
        };
        name == first
            || (alone
                && self
                    .parts
                    .iter()
                    .any(|p| matches!(p, Part::Key(f) if f.name == name)))
    }
}

/// The name of the ` name=` `rest` opens with: a letter or `_`, then
/// letters, digits, `_`, `/`, `(` or `)` (`tok/s(p50)` is one name).
fn key_at(rest: &str) -> Option<&str> {
    let s = rest.strip_prefix(' ')?;
    let end = s.find(|c: char| !(c.is_alphanumeric() || matches!(c, '_' | '/' | '(' | ')')))?;
    let name = &s[..end];
    (s[end..].starts_with('=') && name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_'))
        .then_some(name)
}

/// Whether `parts` read `rest` to its end, each value pushed onto `values`
/// as printed (a set flag as empty). An optional part is tried present
/// first, then left out; a value takes the lengths [`value_ends`] gives, in
/// its order — the way records.py's pattern backtracks.
fn walk(parts: &[Part], rest: &str, values: &mut Vec<(&'static str, String)>) -> bool {
    let Some((p, after)) = parts.split_first() else {
        return rest.is_empty();
    };
    match *p {
        Part::Lit(t) => rest.strip_prefix(t).is_some_and(|r| walk(after, r, values)),
        Part::Flag(n) => {
            if let Some(r) = rest.strip_prefix(' ').and_then(|r| r.strip_prefix(n)) {
                values.push((n, String::new()));
                if walk(after, r, values) {
                    return true;
                }
                values.pop();
            }
            walk(after, rest, values)
        }
        Part::Key(f) | Part::Pos(f) => {
            let at = match p {
                Part::Key(_) => rest
                    .strip_prefix(' ')
                    .and_then(|r| r.strip_prefix(f.name))
                    .and_then(|r| r.strip_prefix('=')),
                _ => Some(rest),
            };
            if let Some(v) = at {
                for end in value_ends(f.ty, v) {
                    values.push((f.name, v[..end].to_owned()));
                    if walk(after, &v[end..], values) {
                        return true;
                    }
                    values.pop();
                }
            }
            f.opt && walk(after, rest, values)
        }
    }
}

/// The lengths a value of `ty` may take at the start of `rest`, each at a
/// char boundary, in the order records.py's pattern tries them: text the
/// shortest first (it runs up to what follows), a list up to its `]`, any
/// other the longest first within the run up to whitespace.
fn value_ends(ty: Ty, rest: &str) -> Vec<usize> {
    let ends =
        |s: &str| -> Vec<usize> { s.char_indices().map(|(i, c)| i + c.len_utf8()).collect() };
    match ty {
        Ty::Text => ends(rest),
        Ty::List | Ty::Csv => {
            if rest.starts_with('[') {
                rest.find(']').map(|i| vec![i + 1]).unwrap_or_default()
            } else {
                Vec::new()
            }
        }
        _ => {
            let run = &rest[..rest.find(char::is_whitespace).unwrap_or(rest.len())];
            let mut out: Vec<usize> = ends(run)
                .into_iter()
                .filter(|&e| fits(ty, &run[..e]))
                .collect();
            out.reverse();
            out
        }
    }
}

/// Whether `s` is a whole value of `ty` as records.py's pattern reads one.
fn fits(ty: Ty, s: &str) -> bool {
    let digits = |d: &str| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit());
    match ty {
        Ty::U64 => digits(s),
        Ty::I64 => digits(s.strip_prefix('-').unwrap_or(s)),
        Ty::F64(_) => {
            let m = s.strip_prefix('-').unwrap_or(s);
            m == "inf"
                || m == "NaN"
                || m.split_once('.')
                    .map_or_else(|| digits(m), |(i, f)| digits(i) && digits(f))
        }
        Ty::Bool => s == "true" || s == "false",
        Ty::Word => !s.is_empty() && !s.contains(char::is_whitespace),
        Ty::Text => !s.is_empty(),
        Ty::List | Ty::Csv => {
            s.len() >= 2
                && s.starts_with('[')
                && s.ends_with(']')
                && !s[1..s.len() - 1].contains(']')
        }
    }
}

/// One record a [`Log`] read whole: its kind, its line and the line's
/// number, and the values its parts carry, as printed. Each read names the field; one the kind does
/// not have, one of another type than asked, an optional one the line left
/// out (read as present) and a value past its type's range are refused by
/// name.
#[derive(Clone, Debug)]
pub struct Fields {
    kind: &'static Kind,
    at: usize,
    line: String,
    values: Vec<(&'static str, String)>,
}

impl Fields {
    #[must_use]
    pub fn kind(&self) -> &'static Kind {
        self.kind
    }

    /// The line as printed.
    #[must_use]
    pub fn line(&self) -> &str {
        &self.line
    }

    /// The line's number in the log, from 1.
    #[must_use]
    pub fn at(&self) -> usize {
        self.at
    }

    /// The value part `name`, checked to be of a type `is` takes, and its
    /// text when the line carries it.
    fn get(
        &self,
        name: &str,
        asked: &'static str,
        is: fn(Ty) -> bool,
    ) -> Result<(&'static str, Option<&str>), ReadError> {
        let f = self
            .kind
            .parts
            .iter()
            .find_map(|p| match p {
                Part::Key(f) | Part::Pos(f) if f.name == name => Some(*f),
                _ => None,
            })
            .ok_or_else(|| ReadError::NoField {
                kind: self.kind.name,
                field: name.to_owned(),
            })?;
        if !is(f.ty) {
            return Err(ReadError::Type {
                kind: self.kind.name,
                field: f.name,
                ty: f.ty,
                asked,
            });
        }
        let v = self
            .values
            .iter()
            .find(|(n, _)| *n == f.name)
            .map(|(_, v)| v.as_str());
        Ok((f.name, v))
    }

    /// [`Fields::get`]'s text, refused when the line left the part out.
    fn need<'a>(&self, (field, v): (&'static str, Option<&'a str>)) -> Result<&'a str, ReadError> {
        v.ok_or_else(|| ReadError::Missing {
            kind: self.kind.name,
            field,
            line: self.line.clone(),
        })
    }

    /// `text` as a number, refused when it is past the type's range.
    fn number<T: std::str::FromStr>(
        &self,
        field: &'static str,
        text: &str,
    ) -> Result<T, ReadError> {
        text.parse().map_err(|_| ReadError::Value {
            kind: self.kind.name,
            field,
            value: text.to_owned(),
        })
    }

    /// An unsigned integer part.
    pub fn u64(&self, name: &str) -> Result<u64, ReadError> {
        let got = self.get(name, "u64", |t| t == Ty::U64)?;
        self.number(got.0, self.need(got)?)
    }

    /// An optional unsigned integer part, `None` when the line left it out.
    pub fn opt_u64(&self, name: &str) -> Result<Option<u64>, ReadError> {
        let (field, v) = self.get(name, "u64", |t| t == Ty::U64)?;
        v.map(|t| self.number(field, t)).transpose()
    }

    /// A signed integer part.
    pub fn i64(&self, name: &str) -> Result<i64, ReadError> {
        let got = self.get(name, "i64", |t| t == Ty::I64)?;
        self.number(got.0, self.need(got)?)
    }

    /// A float part.
    pub fn f64(&self, name: &str) -> Result<f64, ReadError> {
        let got = self.get(name, "f64", |t| matches!(t, Ty::F64(_)))?;
        self.number(got.0, self.need(got)?)
    }

    /// A word part.
    pub fn word(&self, name: &str) -> Result<&str, ReadError> {
        let got = self.get(name, "word", |t| t == Ty::Word)?;
        self.need(got)
    }

    /// An optional word part, `None` when the line left it out.
    pub fn opt_word(&self, name: &str) -> Result<Option<&str>, ReadError> {
        Ok(self.get(name, "word", |t| t == Ty::Word)?.1)
    }

    /// A text part: the words up to the next part, spaces and all.
    pub fn text(&self, name: &str) -> Result<&str, ReadError> {
        let got = self.get(name, "text", |t| t == Ty::Text)?;
        self.need(got)
    }

    /// A `true`/`false` part.
    pub fn bool(&self, name: &str) -> Result<bool, ReadError> {
        let got = self.get(name, "bool", |t| t == Ty::Bool)?;
        Ok(self.need(got)? == "true")
    }

    /// A `[a,b,c]` part's items, trimmed; `[]` none.
    pub fn csv(&self, name: &str) -> Result<Vec<&str>, ReadError> {
        let got = self.get(name, "csv", |t| t == Ty::Csv)?;
        let inner = self.need(got)?;
        let inner = inner[1..inner.len() - 1].trim();
        Ok(if inner.is_empty() {
            Vec::new()
        } else {
            inner.split(',').map(str::trim).collect()
        })
    }

    /// Whether the flag `name` is set.
    pub fn flag(&self, name: &str) -> Result<bool, ReadError> {
        if !self
            .kind
            .parts
            .iter()
            .any(|p| matches!(p, Part::Flag(n) if *n == name))
        {
            return Err(ReadError::NoField {
                kind: self.kind.name,
                field: name.to_owned(),
            });
        }
        Ok(self.values.iter().any(|(n, _)| *n == name))
    }
}

// ---------------------------------------------------------------- the kinds

use Ty::{Bool, Csv, F64, I64, List, Text, U64, Word};

/// Where the placement puts the experts, before the load.
pub static PLAN: Kind = Kind {
    name: "plan",
    head: "plan",
    doc: "The placement the engine is about to load by: its card, context, where the experts sit, and each plan card's device (role:name:ordinal:usable bytes; by-name for a card a census-free plan opens by its name) under the CUDA enumeration order in force, with the card's free bytes at plan time when the census read them.",
    parts: &[
        key("place", Word, ""),
        key("card", Word, ""),
        key("ctx_max", U64, "positions"),
        key("card_experts", U64, "experts"),
        lit(" ("),
        pos("card_expert_bytes", U64, "B"),
        lit(" B)"),
        key("host_experts", U64, "experts"),
        lit(" ("),
        pos("host_expert_bytes", U64, "B"),
        lit(" B)"),
        key("host_shadow", U64, "B"),
        lit(" B"),
        key("n_l", Word, "experts"),
        lit(" on "),
        pos("n_l_layers", U64, "layers"),
        lit(" layers"),
        key("card_budget", Word, "B"),
        opt("card_free", U64, "B"),
        opt("devices", Csv, ""),
        opt("cuda_order", Word, ""),
    ],
};

/// Where a qwen3moe-family placement puts the routed experts, before the
/// load.
pub static PLAN38: Kind = Kind {
    name: "plan38",
    head: "plan",
    doc: "A qwen3moe-family placement the engine is about to load by: its card, the architecture (a placed qwen3moe or qwen35moe file names it; a qwen4exp line does not), the expert rule (host or card), the context, where the routed experts sit (each layer's id prefix on the card, a placed line stating as n_l the fewest and the most one layer keeps; under plan (b′) the expert tier card and the next ids it holds), each plan card's device as the plan record names them, the card's free bytes at plan time when the census read them, and — on the placed plan a run with --place unset took because the whole file did not fit the free bytes — why it did not; for a file with a row-gathered table (a qwen4exp file's PLE table), the tier the plan reads it from (rows, host or nvme) and the host room's reading that chose it (read).",
    parts: &[
        key("place", Word, ""),
        key("card", Word, ""),
        opt("arch", Word, ""),
        key("experts", Word, ""),
        key("ctx_max", U64, "positions"),
        key("host_experts", U64, "experts"),
        key("card_experts", U64, "experts"),
        opt("n_l", Word, "experts"),
        opt("tier", Word, ""),
        opt("tier_experts", U64, "experts"),
        opt("card_free", U64, "B"),
        opt("why", Word, ""),
        opt("rows", Word, ""),
        opt("read", Word, ""),
        opt("devices", Csv, ""),
        opt("cuda_order", Word, ""),
    ],
};

/// The loaded model.
pub static LOAD: Kind = Kind {
    name: "load",
    head: "load",
    doc: "The load: device bytes, the ring shadows' host bytes, the cards it loaded (an expert tier card's experts and resident bytes), the context, and the modes and levers the body holds.",
    parts: &[
        key("resident_bytes", U64, "B"),
        key("shadow", Word, ""),
        lit(" "),
        pos("shadow_bytes", U64, "B"),
        key("unified_addressing", I64, ""),
        key("cards", Csv, ""),
        opt("tier_experts", U64, "experts"),
        opt("tier_bytes", U64, "B"),
        key("ctx", U64, "positions"),
        key("layers", U64, ""),
        key("top_k", U64, ""),
        key("mode", Word, ""),
        key("place", Word, ""),
        key("pin_main", Word, ""),
        key("pinned", Bool, ""),
        key("prefill", Word, ""),
        key("ced", Text, ""),
        key("group", U64, "batches"),
        lit(" in "),
        pos("load_s", F64(1), "s"),
        lit(" s (runtime value)"),
    ],
};

/// The loaded model, as `generate::Generator` writes it.
pub static LOAD_GENERATOR: Kind = Kind {
    name: "load_generator",
    head: "load",
    doc: "The load a decode loop opened: device bytes, the context, the body's fields, the cards it loaded (a V4.1 or GLM-5.3 binary's; an expert tier card's experts and resident bytes), the prompt feed, a GLM-5.3 load's prompt group and whether its host union's row-lane packs prefetch, the step mode and the pin.",
    parts: &[
        key("resident_bytes", U64, "B"),
        key("ctx", U64, "positions"),
        key("layers", U64, ""),
        key("top_k", U64, ""),
        key("shadow", Word, ""),
        lit(" "),
        pos("shadow_bytes", U64, "B"),
        key("unified_addressing", I64, ""),
        opt("cards", Csv, ""),
        opt("tier_experts", U64, "experts"),
        opt("tier_bytes", U64, "B"),
        opt("prefill", Word, ""),
        opt("group", U64, "batches"),
        opt("lane_prefetch", Word, ""),
        key("mode", Word, ""),
        key("place", Word, ""),
        key("pin_main", Word, ""),
        key("pinned", Bool, ""),
        lit(" in "),
        pos("load_s", F64(1), "s"),
        lit(" s (runtime value)"),
    ],
};

/// The load's phases, each the wall of its span inside the load line's
/// `load_s`: the file's headers, the placement plan when its loader timed
/// it, the card's context and device modules, the weights upload, the
/// derived weights, the plan's host set, the body's build, the output head,
/// and the rest of the span.
pub static LOAD_PHASES: Kind = Kind {
    name: "load_phases",
    head: "load phases",
    doc: "The load's phases, each the wall of its span inside the load line's load_s: the file's headers, the placement plan when its loader timed it, the card's context and device modules, the weights upload, the derived weights, the plan's host set, the body's build, the output head, and the rest of the span.",
    parts: &[
        key("open_s", F64(2), "s"),
        opt("plan_s", F64(2), "s"),
        key("context_s", F64(2), "s"),
        key("upload_s", F64(2), "s"),
        key("derive_s", F64(2), "s"),
        key("host_set_s", F64(2), "s"),
        key("body_s", F64(2), "s"),
        key("head_s", F64(2), "s"),
        key("other_s", F64(2), "s"),
    ],
};

/// The host set read in at load.
pub static HOST_POPULATE: Kind = Kind {
    name: "host_populate",
    head: "host_populate=",
    doc: "The plan's host set read in at load (MADV_POPULATE_READ), the wall it took, and the part of it in the r8 sidecar.",
    parts: &[
        pos("host_populate_bytes", U64, "B"),
        lit(" in "),
        pos("populate_s", F64(1), "s"),
        lit(" s (runtime value)"),
        key("sidecar_bytes", U64, "B"),
    ],
};

/// The host set not read in.
pub static HOST_POPULATE_OFF: Kind = Kind {
    name: "host_populate_off",
    head: "host_populate=off",
    doc: "The host set was not read in at load (BLOOMERY_HOST_POPULATE=0).",
    parts: &[],
};

/// The host set locked.
pub static HOST_LOCK: Kind = Kind {
    name: "host_lock",
    head: "host_lock=",
    doc: "The host set mlocked for the model's life (BLOOMERY_HOST_LOCK=1), and the part of it in the r8 sidecar.",
    parts: &[
        pos("host_lock_bytes", U64, "B"),
        lit(" B"),
        key("sidecar_bytes", U64, "B"),
    ],
};

/// The DSpark draft's load.
pub static LOAD_DRAFT: Kind = Kind {
    name: "load_draft",
    head: "load",
    doc: "The DSpark draft's load: its card, width, the target's tapped layers, the device bytes it took, the reserve the placement made for it on a tier card, and the draft card's free bytes.",
    parts: &[
        key("draft", Word, ""),
        key("card", Word, ""),
        key("width", U64, "positions"),
        key("target_layers", List, ""),
        key("feature_width", U64, ""),
        key("resident", U64, "B"),
        opt("reserve", U64, "B"),
        key("draft_card_free", U64, "B"),
        lit(" of "),
        pos("draft_card_total", U64, "B"),
        lit(" in "),
        pos("load_s", F64(1), "s"),
        lit(" s (runtime value)"),
    ],
};

/// A qwen4exp run of `generate_qwen3moe` that drafts nothing
/// ([`bloomery_levers::Draft38Off`]).
pub static LOAD_DRAFT_OFF38: Kind = Kind {
    name: "load_draft_off38",
    head: "load draft=off",
    doc: "A qwen4exp run that drafts nothing, after its load line, and why: BLOOMERY_DRAFT=off, or \
          unset and the condition that left the plain path (--place gate, --logits, the route \
          trace, no file where the MTP draft would be opened, or the last window past --ctx).",
    parts: &[lit(" ("), pos("why", Text, ""), lit(")")],
};

/// A drafted qwen4exp load's MTP head
/// (`model::arch::qwen35moe::head_list::head_rows_of`).
pub static MTP_HEAD38: Kind = Kind {
    name: "mtp_head38",
    head: "mtp head",
    doc: "A drafted qwen4exp load's MTP head, after its load draft=mtp line: list or full, the \
          rows it scores, what picked it (shipped: BLOOMERY_MTP_HEAD_ROWS unset and the target's \
          tokenizer the shipped list's; other-tokenizer: unset on a target of another tokenizer, \
          the full head; set: the lever's value) and why.",
    parts: &[
        key("head", Word, ""),
        key("rows", U64, ""),
        key("from", Word, ""),
        key("why", Text, ""),
    ],
};

/// The drafted Qwen3.8 seat's choice between a kept prefix and the draft.
pub static MTP_KEEP38: Kind = Kind {
    name: "mtp_keep38",
    head: "mtp keep",
    doc: "bloomery-serve-qwen38 under the MTP draft: a request whose kept prefix leaves the draft \
          off (a cut, or the draft already off), and the branch it took: kept (the prefix at or \
          past the break-even; the draft proposes nothing from there) or reset (below it: the \
          prompt prefilled whole from position 0, the draft on), the prefix the keep rule \
          granted, the break-even and the reply tokens it was derived for.",
    parts: &[
        key("branch", Word, ""),
        key("prefix", U64, "positions"),
        key("break_even", U64, "positions"),
        key("reply", U64, "tokens"),
    ],
};

/// The step's capture.
pub static CAPTURE: Kind = Kind {
    name: "capture",
    head: "capture",
    doc: "The decode step captured before the prompt: its graph's nodes.",
    parts: &[key("graph_nodes", U64, "nodes")],
};

/// The pair pass's capture.
pub static CAPTURE_PAIR: Kind = Kind {
    name: "capture_pair",
    head: "capture",
    doc: "The two-row pass captured before the prompt: its graph's nodes.",
    parts: &[key("pair_graph_nodes", U64, "nodes")],
};

/// The prompt batch's buffers.
pub static PREFILL_BYTES: Kind = Kind {
    name: "prefill_bytes",
    head: "prefill",
    doc: "The prompt batch's device bytes, of them the batch-wide projections' and what the batches past a group's first hold.",
    parts: &[
        key("batch_bytes", U64, "B"),
        key("proj_bytes", U64, "B"),
        key("group", U64, "batches"),
        key("group_bytes", U64, "B"),
    ],
};

/// A GLM-5.3 prompt batch's device bytes, as `generate_glm5next` writes them.
pub static PROMPT_UNITS: Kind = Kind {
    name: "prompt_units",
    head: "prefill units",
    doc: "A GLM-5.3 prompt batch's device bytes once made: the buffers a group's units share, one unit's own, the units made (one a batch of a group), the host sums', and the card's free bytes after them. The plan reserves nothing for a group's units past the first: they come out of the card's margin.",
    parts: &[
        key("shared_bytes", U64, "B"),
        key("unit_bytes", U64, "B"),
        key("units", U64, ""),
        key("hsum_bytes", U64, "B"),
        key("free_bytes", U64, "B"),
    ],
};

/// A prompt call's plan.
pub static CALL_PLAN: Kind = Kind {
    name: "call_plan",
    head: "call plan",
    doc: "A prompt call's plan: its positions, batches and groups, the sizes it cuts by, the ring the triangle reads, and the triangle's state.",
    parts: &[
        key("first", U64, "positions"),
        key("end", U64, "positions"),
        key("batches", U64, ""),
        key("group", U64, "batches"),
        key("groups", U64, ""),
        key("t_max", U64, "positions"),
        key("chunk", U64, "positions"),
        key("sub_chunks", U64, "chunks"),
        key("ring", U64, "positions"),
        key("layers", U64, ""),
        key("ced", Text, ""),
    ],
};

/// A call's groups under one group lever.
pub static CALL_GROUPS: Kind = Kind {
    name: "call_groups",
    head: "call groups",
    doc: "The batches each group of the call holds under BLOOMERY_PREFILL_GROUP=g.",
    parts: &[key("g", U64, "batches"), key("sizes", Csv, "batches")],
};

/// One batch of a call.
pub static CALL_BATCH: Kind = Kind {
    name: "call_batch",
    head: "call batch",
    doc: "One batch of the call: its positions, its group and place in it under the lever, and its chunks' bounds.",
    parts: &[
        key("b", U64, ""),
        key("first", U64, "positions"),
        key("end", U64, "positions"),
        key("group", U64, ""),
        key("set", U64, ""),
        key("chunks", U64, ""),
        key("cuts", Csv, "positions"),
    ],
};

/// One layer's facts the call's launches follow.
pub static CALL_LAYER: Kind = Kind {
    name: "call_layer",
    head: "call layer",
    doc: "One layer's facts the call's launches follow: its compression ratio, what it owns and runs, and the experts its card holds.",
    parts: &[
        key("l", U64, ""),
        key("ratio", U64, ""),
        key("compressor", Bool, ""),
        key("gated", Bool, ""),
        key("index_keys", Bool, ""),
        key("indexer", Bool, ""),
        key("engram", Bool, ""),
        key("card_experts", U64, "experts"),
    ],
};

/// The call's triangle.
pub static CALL_NEED: Kind = Kind {
    name: "call_need",
    head: "call need",
    doc: "The call's needs: each layer's block and latent starts, and the positions they run over every layer.",
    parts: &[
        key("features_from", U64, "positions"),
        key("block_positions", U64, "positions"),
        key("part_positions", U64, "positions"),
        key("full_from", Csv, "positions"),
        key("part_from", Csv, "positions"),
    ],
};

/// One layer-batch of a call.
pub static CALL_LB: Kind = Kind {
    name: "call_lb",
    head: "call lb",
    doc: "One layer-batch of the call: the chunk its latent part and its block start at, and the sub-blocks the projections run over each.",
    parts: &[
        key("b", U64, ""),
        key("layer", U64, ""),
        key("run", U64, "chunks"),
        key("full", U64, "chunks"),
        key("part_sb", Csv, "chunks"),
        key("full_sb", Csv, "chunks"),
    ],
};

/// The finite probe on.
pub static CHECK_FINITE: Kind = Kind {
    name: "check_finite",
    head: "check_finite=on:",
    doc: "BLOOMERY_CHECK_FINITE=1: every position runs through the finite probe first.",
    parts: &[lit(
        " every position observed eagerly, taken back, then stepped through the engine",
    )],
};

/// The fed ids.
pub static FED: Kind = Kind {
    name: "fed",
    head: "fed",
    doc: "The ids fed before the first generated token: how many, the first and last four, and where the depth sequence starts.",
    parts: &[
        key("ids", U64, "positions"),
        key("first", List, ""),
        key("last", List, ""),
        key("depth_sequence_from", U64, "positions"),
    ],
};

/// One arm of an `--arm` list.
pub static ARM: Kind = Kind {
    name: "arm",
    head: "arm",
    doc: "An arm of one load's --arm list, before its lines: its index, the list's length, its feed (lcg, a corpus, or repeat: a --repeat run of the prompt flags), its fed ids and its generated count.",
    parts: &[
        key("i", U64, ""),
        key("arms", U64, ""),
        key("feed", Word, ""),
        key("ids", U64, "positions"),
        key("n", U64, "tokens"),
    ],
};

/// The head's last logits row, by its bits.
pub static LOGITS: Kind = Kind {
    name: "logits",
    head: "logits",
    doc: "--logits: the head's last logits row after the loop, its length, argmax and FNV-1a 64 of its f32 bits.",
    parts: &[
        key("n", U64, ""),
        key("argmax", U64, ""),
        key("fnv64", Word, ""),
    ],
};

/// The logits row behind one generated token, by its two largest entries.
pub static TOP2: Kind = Kind {
    name: "top2",
    head: "top2",
    doc: "--top2 K: the logits row behind each of the first K generated tokens (i 0 the prompt call's last row), its two largest entries' ids and values, the larger first, and their margin.",
    parts: &[
        key("i", U64, ""),
        key("top1", U64, ""),
        key("top1_logit", F64(4), ""),
        key("top2", U64, ""),
        key("top2_logit", F64(4), ""),
        key("margin", F64(4), ""),
    ],
};

/// `--rows`: the rows `--top2` read, appended whole to a file.
pub static ROWS: Kind = Kind {
    name: "rows",
    head: "rows",
    doc: "--rows FILE: a run's --top2 rows appended whole to FILE as little-endian f32, after the run's top2 records: the rows, each row's length, the bytes written and FILE.",
    parts: &[
        key("rows", U64, ""),
        key("n", U64, ""),
        key("bytes", U64, "B"),
        key("path", Text, ""),
    ],
};

/// `--ignore-eos`: the ids the pick passes over.
pub static IGNORE_EOS: Kind = Kind {
    name: "ignore_eos",
    head: "ignore eos",
    doc: "--ignore-eos: the vocabulary's end-of-generation ids, which the pick passes over as the serve does under a request's ignore_eos and --top2 does not rank.",
    parts: &[key("ids", Csv, "")],
};

/// `--dump-table`: the card's map as a run's token 0 used it.
pub static TABLE_DUMP: Kind = Kind {
    name: "table_dump",
    head: "table dump",
    doc: "--dump-table FILE: the stage card's copy of the slot map as the second run's token 0 ran on it, written to FILE: its entries, those on the card, those that differ from the host map, whether the read after Target::reset (FILE.before) is the same, and FILE.",
    parts: &[
        key("entries", U64, ""),
        key("on_card", U64, ""),
        key("vs_map", U64, ""),
        key("same_before", Bool, ""),
        key("path", Text, ""),
    ],
};

/// `--card-table`: the fixed placement a load made from a dumped table.
pub static CARD_TABLE: Kind = Kind {
    name: "card_table",
    head: "card table",
    doc: "--card-table FILE: a load placed by a dumped table, residency off: its layers, its card's experts, the entries whose card or host side differs from FILE's, those on the card in both whose slot differs (the load puts each layer's in id order), and FILE.",
    parts: &[
        key("layers", U64, ""),
        key("on_card", U64, ""),
        key("set_diff", U64, ""),
        key("slot_diff", U64, ""),
        key("path", Text, ""),
    ],
};

/// `--table`: the stage card's copy of the slot map, read back.
pub static SLOT_TABLE: Kind = Kind {
    name: "slot_table",
    head: "slot table",
    doc: "--table: the stage card's copy of the slot map read back after the runs: its layers and entries, the entries on the card, whether the load runs the residency machine, the entries that differ from the host map, from the ledger's live and landing slots (0 without a machine), and from the map before the first run, the card slots two ids of one layer both name, and the first entries that differ from the map or the ledger (l<layer>/e<id>:card=,map=,ledger=, h the host mark; none when none).",
    parts: &[
        key("layers", U64, ""),
        key("entries", U64, ""),
        key("on_card", U64, ""),
        key("machine", Bool, ""),
        key("vs_map", U64, ""),
        key("vs_ledger", U64, ""),
        key("vs_load", U64, ""),
        key("doubled", U64, ""),
        key("first", Text, ""),
    ],
};

/// Generated token 0.
pub static STEP0: Kind = Kind {
    name: "step0",
    head: "step 0",
    doc: "Generated token 0, out of the last fed position, and the feed's wall.",
    parts: &[
        lit(" "),
        pos("pos", U64, ""),
        lit(" "),
        pos("token", U64, ""),
        lit(" (the "),
        pos("fed", U64, "positions"),
        lit(" fed steps in "),
        pos("feed_s", F64(1), "s"),
        lit(" s, runtime value)"),
    ],
};

/// The host tier's batch services.
pub static STAT_PREFILL: Kind = Kind {
    name: "stat_prefill",
    head: "stat prefill",
    doc: "The host tier's batch services since load: layers, columns, host slots and the union calls' wall.",
    parts: &[
        key("union_layers", U64, ""),
        key("union_cols", U64, ""),
        key("union_host_slots", U64, ""),
        key("union_ms", F64(1), "ms"),
    ],
};

/// The last call's triangle.
pub static STAT_PREFILL_CED: Kind = Kind {
    name: "stat_prefill_ced",
    head: "stat prefill",
    doc: "The triangle's state and the last call's needs: each layer's block and latent starts.",
    parts: &[
        key("ced", Text, ""),
        key("first", U64, "positions"),
        key("end", U64, "positions"),
        key("features_from", U64, "positions"),
        key("block_positions", U64, "positions"),
        key("part_positions", U64, "positions"),
        key("full_from", Csv, "positions"),
        key("part_from", Csv, "positions"),
    ],
};

/// Where a prompt's batches spent their time.
pub static STAT_PREFILL_SPLIT: Kind = Kind {
    name: "stat_prefill_split",
    head: "stat prefill split",
    doc: "The prompt batches' host time by phase, summed and per layer-batch (_lb), the queue entries a layer-batch enqueues, and with card timing each layer-batch's card time.",
    parts: &[
        key("group", U64, "batches"),
        key("batches", U64, ""),
        key("layer_batches", U64, ""),
        key("prologue_ms", F64(1), "ms"),
        key("chain_ms", F64(1), "ms"),
        key("union_ms", F64(1), "ms"),
        key("wait_ms", F64(1), "ms"),
        key("enqueue_ms", F64(1), "ms"),
        key("copy_ms", F64(1), "ms"),
        key("union_lb", F64(2), "ms/lb"),
        key("wait_lb", F64(2), "ms/lb"),
        key("wait_first_lb", F64(2), "ms/lb"),
        key("enqueue_lb", F64(2), "ms/lb"),
        key("copy_lb", F64(2), "ms/lb"),
        key("entries_route", F64(1), "entries/lb"),
        key("entries_shadow", F64(1), "entries/lb"),
        key("excluded_lb", F64(1), "slots/lb"),
        opt("card", Word, ""),
        opt("card_out_ms", F64(1), "ms"),
        opt("card_in_ms", F64(1), "ms"),
        opt("card_proj_ms", F64(1), "ms"),
        opt("card_out_lb", F64(2), "ms/lb"),
        opt("card_in_lb", F64(2), "ms/lb"),
        opt("card_proj_lb", F64(2), "ms/lb"),
    ],
};

/// One batch's first steps, counted.
pub static STAT_PREFILL_FRONT: Kind = Kind {
    name: "stat_prefill_front",
    head: "stat prefill front",
    doc: "The queue entries one batch's first steps enqueued: its chunks' gathers and its embedding broadcast.",
    parts: &[key("b", U64, ""), key("entries", U64, "entries")],
};

/// One layer-batch, counted.
pub static STAT_PREFILL_LB: Kind = Kind {
    name: "stat_prefill_lb",
    head: "stat prefill lb",
    doc: "The queue entries one layer-batch enqueued: its route (with its upload, join and tap) and its shadow; block says whether it ran one. With card timing, its card time (card_out: its first launch to its route's copies; card_in: its shadow, where it has a block) and its serve's host time (union, and the wait on its route's copies), which the split line's are the sums of.",
    parts: &[
        key("b", U64, ""),
        key("layer", U64, ""),
        key("block", Bool, ""),
        key("entries_route", U64, "entries"),
        key("entries_shadow", U64, "entries"),
        opt("card_out_ms", F64(2), "ms"),
        opt("card_in_ms", F64(2), "ms"),
        opt("union_ms", F64(2), "ms"),
        opt("wait_ms", F64(2), "ms"),
    ],
};

/// The prompt feed's wall.
/// `clef_hidden`'s prompt call: the positions it fed, the model's width, the
/// hidden-state bytes it wrote, and the call's wall from the first id's
/// upload to the last row's readback.
pub static CLEF_HIDDEN: Kind = Kind {
    name: "clef_hidden",
    head: "clef hidden",
    doc: "The prompt call's final-norm hidden states: positions fed, the model's width, the f32 bytes written, and the call's wall from the ids' upload to the last row's readback.",
    parts: &[
        key("n", U64, "positions"),
        key("width", U64, ""),
        key("bytes", U64, "B"),
        key("ms", F64(1), "ms"),
        key("tok/s", F64(1), "tok/s"),
    ],
};

/// `clef_hidden`'s output-embedding rows: the ids read, the width, and the
/// f32 bytes written.
pub static CLEF_ROWS: Kind = Kind {
    name: "clef_rows",
    head: "clef rows",
    doc: "The output embedding's rows for the given ids, dequantized on the host: ids read, the width, and the f32 bytes written.",
    parts: &[
        key("ids", U64, ""),
        key("width", U64, ""),
        key("bytes", U64, "B"),
    ],
};

/// The NVMe expert tier's arena over one paged load: its counters as the
/// tier holds them (`NvTierStats`) — the misses `ensure` read, the slots
/// filled and their bytes and wall, the evictions, the resident bytes (at
/// or under the budget at every step), the reads the page cache served, and
/// the drops of the mapping's pages a read brought in, their bytes and wall.
pub static NVTIER: Kind = Kind {
    name: "nvtier",
    head: "nvtier",
    doc: "The NVMe expert tier's RAM arena on a paged load: the arena's budget, the paged experts it serves, the misses ensure read, the slots filled and their bytes and wall, the evictions, the resident bytes, the reads the page cache served, and the drops of the model mapping's pages a read of the arena's ids brought in (the calls, the whole-page bytes named to the kernel, the wall on the reader's thread).",
    parts: &[
        key("budget", U64, "B"),
        key("paged", U64, "B"),
        key("misses", U64, ""),
        key("fills", U64, ""),
        key("fill_bytes", U64, "B"),
        key("fill_ns", U64, "ns"),
        key("evictions", U64, ""),
        key("resident_bytes", U64, "B"),
        key("buffered_reads", U64, ""),
        key("buffered_bytes", U64, "B"),
        key("drops", U64, ""),
        key("drop_bytes", U64, "B"),
        key("drop_ns", U64, "ns"),
    ],
};

/// One span's tier census on a paged load (`bloomery_gpu::host::census`): the
/// span's phase, its tokens, the routed picks its host services saw and where
/// each was served — the stage card (or a tier card), the NVMe tier's arena,
/// the model file's mapping — the bytes it moved between the tiers (the
/// arena's fills, the residency machine's promotions onto the card and its
/// demotions to the host), the tier's drops of the mapping's pages and
/// their wall, and the pool's dispatches that waited for another caller's
/// job and their wait.
pub static TIER_CENSUS: Kind = Kind {
    name: "tier_census",
    head: "tier census",
    doc: "One span of a paged load: its phase (a gate's prompt, steps or slots phase; a server's call), the tokens its host services carried, the routed picks they saw and of them those served by a card (the stage card, a tier card, or a prompt's host slots left to the card route), read from the NVMe tier's RAM arena, or read through the model file's mapping (card + arena + file = picks when every host slot was read once); the bytes the arena's fills read, the bytes of the experts the residency machine put on the stage card and sent back to the host; the tier's drops of the mapping's pages and their wall on the dropper's thread; and the worker pool's dispatches that waited for another caller's job, with their wait.",
    parts: &[
        key("phase", Word, ""),
        key("tokens", U64, ""),
        key("picks", U64, ""),
        key("card", U64, ""),
        key("arena", U64, ""),
        key("file", U64, ""),
        key("fill_bytes", U64, "B"),
        key("promote_bytes", U64, "B"),
        key("demote_bytes", U64, "B"),
        key("drops", U64, ""),
        key("drop_ns", U64, "ns"),
        key("dispatch_waits", U64, ""),
        key("dispatch_wait_ns", U64, "ns"),
    ],
};

/// One result row of `probe_nvread`: an arm over a case of cold routed-expert
/// runs, the batches' rate quantiles and what the run asked of the drive.
pub static NVREAD: Kind = Kind {
    name: "nvread",
    head: "nvread",
    doc: "One arm (cache: WILLNEED then POPULATE_READ split across threads; direct1 and directN: O_DIRECT preads of each part on one thread and on the same split; drive: in the range case, O_DIRECT preads of each stack's whole contiguous run in large requests on one thread) over one case (k1, k2, k3: that many experts from distinct layers a batch; range: one layer's whole expert range from a named id): the experts a batch reads, the threads, the measured and warm-up batches, the batches redrawn because a dropped run was still resident, the mean useful bytes and the mean page-rounded bytes a batch, every byte the arm asked of the drive, the batch rate's median, 10th and 90th percentile, the median batch wall, the draw seed, whether the run holds the lease, and the cache arm's picks.",
    parts: &[
        key("arm", Word, ""),
        key("case", Word, ""),
        key("experts", U64, ""),
        key("threads", U64, ""),
        key("batches", U64, ""),
        key("warmup", U64, ""),
        key("redraws", U64, ""),
        key("bytes", U64, "B"),
        key("span_bytes", U64, "B"),
        key("read_bytes", U64, "B"),
        key("median", F64(3), "GB/s"),
        key("p10", F64(3), "GB/s"),
        key("p90", F64(3), "GB/s"),
        key("ms", F64(3), "ms"),
        key("seed", U64, ""),
        key("lease", Bool, ""),
        opt("draws", Csv, ""),
    ],
};

pub static TIME_PROMPT: Kind = Kind {
    name: "time_prompt",
    head: "time prompt",
    doc: "The feed's wall, before the first fed step to after the readback of generated token 0, and the passes it took.",
    parts: &[
        key("n", U64, "positions"),
        key("ms", F64(4), "ms"),
        key("tok/s", F64(2), "tok/s"),
        key("passes", U64, ""),
        key("kind", Word, ""),
    ],
};

/// One layer-batch of a prompt's batch walks.
pub static STAT_PROMPT_LB: Kind = Kind {
    name: "stat_prompt_lb",
    head: "stat prompt lb",
    doc: "One layer-batch of a prompt's batch walks (Qwen3.8's ubatches, GLM-5.3's batches) under \
          BLOOMERY_STEP_STATS: its batch in the prompt, its layer, its columns, the \
          host slots its union listed and their per-expert counts (the listed experts, the widest \
          one's columns, the hot ones past an L2-sized activation set and their columns, and Σ m² \
          over the listed experts, which with the slots gives the routing's spread), its serve's \
          host time (the wait on the route's copies; the \
          union call's wall, which carries the routing scan and the plan build in front of it; the \
          serve's whole wall, the upload's enqueue in it), the host wall its front, shared expert \
          and gated sum took to enqueue, and its card time by part (the front's launches, the \
          route's downloads, the card route on a card layer and the shared expert under the union, \
          the sums' upload, the gated sum).",
    parts: &[
        key("b", U64, ""),
        key("layer", U64, ""),
        key("cols", U64, ""),
        key("slots", U64, "slots"),
        key("experts", U64, ""),
        key("m_max", U64, "columns"),
        key("m_hot", U64, "experts"),
        key("cols_hot", U64, "columns"),
        key("m_sq", U64, "columns^2"),
        key("wait_ms", F64(2), "ms"),
        key("union_ms", F64(2), "ms"),
        key("serve_ms", F64(2), "ms"),
        key("enqueue_ms", F64(2), "ms"),
        key("card_front_ms", F64(2), "ms"),
        key("card_down_ms", F64(2), "ms"),
        key("card_shadow_ms", F64(2), "ms"),
        key("card_upload_ms", F64(2), "ms"),
        key("card_back_ms", F64(2), "ms"),
    ],
};

/// Where a prompt's batch walks spent their time.
pub static STAT_PROMPT_SPLIT: Kind = Kind {
    name: "stat_prompt_split",
    head: "stat prompt split",
    doc: "A prompt's batch walks (Qwen3.8's ubatches, GLM-5.3's batches) under \
          BLOOMERY_STEP_STATS, summed: the batches and \
          layer-batches that ran, the host prologue every walk's plan took (Qwen3.8: the PLE \
          rows, the record, their copy) and the walks' whole wall, the host walls the call spent \
          outside the walks reading the fault word and waiting for its checkpoints (GLM-5.3; 0 \
          where the call notes none), the serves' union and wait and the host \
          slots they listed with their per-expert counts (the mean listed experts, hot experts \
          and their columns and Σ m² a layer-batch, and the widest one expert's columns over the \
          prompt), the walks' wall less the serves' as the enqueue's share, and the card \
          time by part summed (the fronts, the route downloads, the shared experts, the uploads, \
          the gated sums) with per layer-batch means.",
    parts: &[
        key("ubatches", U64, ""),
        key("layer_batches", U64, ""),
        key("prologue_ms", F64(1), "ms"),
        key("walk_ms", F64(1), "ms"),
        key("fault_ms", F64(1), "ms"),
        key("ckpt_ms", F64(1), "ms"),
        key("union_ms", F64(1), "ms"),
        key("wait_ms", F64(1), "ms"),
        key("serve_ms", F64(1), "ms"),
        key("enqueue_ms", F64(1), "ms"),
        key("host_slots", U64, "slots"),
        key("union_lb", F64(2), "ms/lb"),
        key("wait_lb", F64(2), "ms/lb"),
        key("serve_lb", F64(2), "ms/lb"),
        key("enqueue_lb", F64(2), "ms/lb"),
        key("slots_lb", F64(1), "slots/lb"),
        key("experts_lb", F64(1), "experts/lb"),
        key("m_max", U64, "columns"),
        key("m_hot_lb", F64(1), "experts/lb"),
        key("cols_hot_lb", F64(1), "columns/lb"),
        key("m_sq_lb", F64(1), "columns^2/lb"),
        key("card_front_ms", F64(1), "ms"),
        key("card_down_ms", F64(1), "ms"),
        key("card_shadow_ms", F64(1), "ms"),
        key("card_upload_ms", F64(1), "ms"),
        key("card_back_ms", F64(1), "ms"),
        key("card_front_lb", F64(2), "ms/lb"),
        key("card_down_lb", F64(2), "ms/lb"),
        key("card_shadow_lb", F64(2), "ms/lb"),
        key("card_upload_lb", F64(2), "ms/lb"),
        key("card_back_lb", F64(2), "ms/lb"),
    ],
};

/// A residency boundary ([`bloomery_gpu::host::swap::PassReport`]).
pub static RESIDENCY_PASS: Kind = Kind {
    name: "residency_pass",
    head: "residency pass",
    doc: "A residency boundary: what the pass before it was (none, step, pair, slots — a pass \
          of resident slots' rows, all kept — slots_drafted — a drafted pass of resident slots' \
          verify rows, each slot's accepted rows kept — prompt — a prompt call's \
          rows are not counted — abandoned, or driver: the machine's own test driver), its passes since the load or the last reset, the rows the pass before it kept (their \
          count, and as a bit mask, bit r: row r kept — a prefix's mask \
          for every pass but a drafted slots pass's), the flips that went live there (late: their copies had not completed and \
          the engine stream waited), the flips the rule made, the flips in flight after it, and \
          the bytes its flips copy; the host's microseconds folding the pass into the rule and in \
          the whole boundary call, of \
          them waiting for the landing jobs' staging and issuing the new flips' copies, and the staging thread's since the last \
          boundary copying into the ring and preparing victims; the experts the machine's thread \
          found not host-resident and read in again since the last boundary (a page the page \
          cache let go), their bytes and its microseconds doing it; the experts it found not \
          host-resident even once read in again since the last boundary (this one's landing \
          victims, the load's and a reset's: each sent to the host all the same, served from the \
          file) and the last of them as layer:expert:site; and the experts sent so that this \
          boundary found still on the host and not resident, which the pass it opens serves from \
          the file, and the first of them as layer:expert:site:passes, the passes each has been \
          served from the file so far; and, when a prompt call ended in the pass before the boundary, the \
          flips it picked (its admits over every layer and unit of the call).",
    parts: &[
        key("pass", Word, ""),
        key("boundary", U64, ""),
        key("kept", U64, ""),
        key("rows", U64, ""),
        key("landed", U64, ""),
        key("late", U64, ""),
        key("made", U64, ""),
        key("in_flight", U64, ""),
        key("bytes", U64, "B"),
        key("end_us", U64, "us"),
        key("boundary_us", U64, "us"),
        key("wait_us", U64, "us"),
        key("issue_us", U64, "us"),
        key("stage_us", U64, "us"),
        key("prepare_us", U64, "us"),
        key("rereads", U64, ""),
        key("reread_bytes", U64, "B"),
        key("reread_us", U64, "us"),
        opt("unresident", U64, ""),
        opt("unresident_last", Csv, ""),
        opt("faulting", U64, ""),
        opt("faulting_first", Csv, ""),
        opt("picked", U64, ""),
    ],
};

/// What `BLOOMERY_RESIDENCY` resolved to ([`bloomery_levers::ResidencyPick`]).
pub static RESIDENCY_LEVER: Kind = Kind {
    name: "residency_lever",
    head: "residency lever",
    doc: "What BLOOMERY_RESIDENCY resolved to before the load: the word the load runs by, and why: \
          set, or unset and what picked it (place: the serving placement's default; off under \
          fixed_place, check_finite, route_trace or prefill_steps, where the machine does not run).",
    parts: &[key("residency", Word, ""), key("why", Word, "")],
};

/// What `BLOOMERY_RESIDENCY` unset resolved to in `generate_qwen3moe`
/// ([`bloomery_levers::Residency38Pick`]).
pub static RESIDENCY_UNSET: Kind = Kind {
    name: "residency_unset",
    head: "residency unset",
    doc: "What BLOOMERY_RESIDENCY unset resolved to in generate_qwen3moe: the word the load runs \
          by (off, or mid-p<P>-s1 on plan (a), P half the plan's fewest card experts a layer) \
          and why, the condition that picked it (the placement, the flags, the file, or the plan: \
          routed experts paged through the NVMe tier's RAM arena, no card expert, no room, or the \
          churn pool past the plan's host headroom or past what MemAvailable leaves).",
    parts: &[key("residency", Word, ""), key("why", Text, "")],
};

/// Adaptive residency's host share at the plan
/// ([`model::placement::churn::ChurnPool`]).
pub static RESIDENCY_HOST: Kind = Kind {
    name: "residency_host",
    head: "residency host",
    doc: "Adaptive residency's host share, from the plan before the load: the lever's word, the \
          seed experts a layer kept on the stage card, the churn pool (the stage card's experts \
          past them, which the load's host set holds too) and its bytes, and the plan's host \
          headroom before and after the pool and any bytes the load hosts beside the plan.",
    parts: &[
        key("residency", Word, ""),
        key("pinned", U64, ""),
        key("churn_experts", U64, ""),
        key("churn_bytes", U64, "B"),
        key("headroom", I64, "B"),
        key("headroom_after", I64, "B"),
    ],
};

/// A residency reset ([`bloomery_gpu::host::swap::ResetReport`]).
pub static RESIDENCY_RESET: Kind = Kind {
    name: "residency_reset",
    head: "residency reset",
    doc: "A residency reset back to the seed: flips in flight cancelled, experts copied back onto \
          the card, entries of the live map that differ from the seed after it (0 when it \
          worked), and the host bytes released for the seed experts back on the card and the \
          victims of cancelled flips, which stay on it.",
    parts: &[
        key("cancelled", U64, ""),
        key("copies", U64, ""),
        key("diff", U64, ""),
        key("dropped_bytes", U64, "B"),
    ],
};

/// A prompt call's pick at one (group, layer)
/// ([`bloomery_gpu::host::swap::CallPick`]).
pub static CALL_STREAM: Kind = Kind {
    name: "call_stream",
    head: "call stream",
    doc: "A prompt call's pick at one layer of one group (host streaming: the residency pool \
          takes the group's hottest host experts): the group, the layer, the experts admitted \
          (each in place of a pool resident sent to the host) and the pool residents kept, the \
          bytes the admitted experts' copies move, the pick's input (FNV-1a 64 of its counts, \
          hex), and the host's microseconds in the pick, of them waiting for the staging thread \
          to take in the call's earlier jobs. The four after, each left out when the pick did \
          not measure it: the microseconds from the pick's start to the staging of the layer's \
          last pick job (the whole copy, read back at the call's end), the least count the pick \
          admitted, the backlog bound it waited on, 1 when the floor came from no probe (a \
          family's caller marks its own; an unmarked one leaves the part out), and 1 when the \
          pick was the machine's refusal (a victim the host could not serve), whose kept 0 is \
          no choice. The two last, on every pick that was not refused: the admitted experts' \
          summed counts (the columns that leave the host union) and the victims' (the columns \
          that move back to it).",
    parts: &[
        key("group", U64, ""),
        key("layer", U64, ""),
        key("admitted", U64, ""),
        key("kept", U64, ""),
        key("bytes", U64, "B"),
        key("counts", Word, ""),
        key("pick_us", U64, "us"),
        key("backlog_us", U64, "us"),
        opt("staged_us", U64, "us"),
        opt("floor", U64, ""),
        opt("backlog", U64, ""),
        opt("fallback", U64, ""),
        opt("refused", U64, ""),
        opt("admit_cols", U64, ""),
        opt("victim_cols", U64, ""),
    ],
};

/// A prompt call's end on the residency machine
/// ([`bloomery_gpu::host::swap::CallReport`]).
pub static CALL_STREAM_END: Kind = Kind {
    name: "call_stream_end",
    head: "call stream end",
    doc: "A prompt call's end on the residency machine: the picks that admitted an expert, the \
          experts admitted and their bytes, the host's microseconds in the picks and of them \
          waiting for the staging thread, whether the call's placement stays for the passes \
          after it (1) or went back to the call's start (0) with the experts copied back, the \
          host's microseconds in the end and, of a return the walk ran before the end, in the \
          walk's return calls, and the experts found not host-resident even once read \
          in again in the call (a pick's victim, whose pick admitted nothing, or an expert the \
          end sent back all the same, served from the file) and the last of them as \
          layer:expert:site.",
    parts: &[
        key("picks", U64, ""),
        key("admitted", U64, ""),
        key("bytes", U64, "B"),
        key("pick_us", U64, "us"),
        key("backlog_us", U64, "us"),
        key("kept", U64, ""),
        key("restored", U64, ""),
        key("end_us", U64, "us"),
        opt("unresident", U64, ""),
        opt("unresident_last", Csv, ""),
        opt("return_us", U64, "us"),
    ],
};

/// One layer of a prompt unit on the expert stream
/// ([`bloomery_gpu::host::xstream::XLayer`]).
pub static XSTREAM: Kind = Kind {
    name: "xstream",
    head: "xstream",
    doc: "One layer of a prompt call's ubatch on the expert stream (`BLOOMERY_XSTREAM=split`): \
          the ubatch, the layer, the ubatch's columns, the host experts it routes there after the \
          residency pick, of them the ones the stream rule sends to the card (`tail`), of those \
          the ones that streamed — the rule's set cut at the ring's half and where the layer's \
          copies would pass the union it keeps, in rank order — and their columns (the serve's \
          excluded slots), the columns the host union keeps, the bytes the stream copies, and the host's microseconds in the split and the \
          issue, of them waiting for the fill threads to take in the staging backlog.",
    parts: &[
        key("ubatch", U64, ""),
        key("layer", U64, ""),
        key("cols", U64, ""),
        key("host", U64, ""),
        key("tail", U64, ""),
        key("streamed", U64, ""),
        key("streamed_columns", U64, ""),
        key("host_columns", U64, ""),
        key("bytes", U64, "B"),
        key("issue_us", U64, "us"),
        key("backlog_us", U64, "us"),
    ],
};

/// A prompt call's end on the expert stream
/// ([`bloomery_gpu::host::xstream::XReport`]).
pub static XSTREAM_END: Kind = Kind {
    name: "xstream_end",
    head: "xstream end",
    doc: "A prompt call's end on the expert stream: the layers that streamed, their experts and \
          bytes, the host's microseconds in the splits and the issues and of them waiting for the \
          staging backlog, the ring's slots a half, the pinned staging's slots (0: the lane \
          copies the source's pageable bytes), the lane's rate the load's probe measured, and the \
          host's microseconds in the end.",
    parts: &[
        key("layers", U64, ""),
        key("streamed", U64, ""),
        key("bytes", U64, "B"),
        key("issue_us", U64, "us"),
        key("backlog_us", U64, "us"),
        key("half_slots", U64, ""),
        key("staging_slots", U64, ""),
        key("lane_gbs", F64(2), "GB/s"),
        key("end_us", U64, "us"),
    ],
};

/// A helper thread the load spawned (`threads::helper::helpers`).
pub static HELPER: Kind = Kind {
    name: "helper",
    head: "helper",
    doc: "A helper thread the load spawned beside the step thread: its name, where it asked to \
          run (pin:<cpu>, sibling:<cpu> — the SMT sibling of that cpu — or float), the cpu it is \
          pinned to (float when none: a refused pin or a core without a sibling floats) and the \
          cpus in its mask once placed.",
    parts: &[
        key("name", Word, ""),
        key("asked", Word, ""),
        key("pinned", Word, ""),
        key("cpus", U64, ""),
    ],
};

/// A dropped residency machine's leak ([`bloomery_gpu::host::swap::Leak`]),
/// on stderr from the drop.
pub static RESIDENCY_LEAK: Kind = Kind {
    name: "residency_leak",
    head: "residency leak",
    doc: "A dropped residency machine kept its staging ring, its staging words and its source \
          rather than free them under a copy or a thread that may still use them: why (join: the \
          staging thread was still inside the source past the deadline; release: the copies \
          waiting on staging could not be let through; drain: the copy stream did not drain \
          within the deadline; fault: the copy stream's query returned a driver error, whose \
          code the code part carries) and the pinned bytes of the ring and the words.",
    parts: &[
        key("reason", Word, ""),
        opt("code", U64, ""),
        key("ring", U64, "B"),
        key("words", U64, "B"),
    ],
};

/// A generated token.
pub static STEP: Kind = Kind {
    name: "step",
    head: "step",
    doc: "Generated token i and the position it is the argmax after.",
    parts: &[
        lit(" "),
        pos("i", U64, ""),
        lit(" "),
        pos("pos", U64, ""),
        lit(" "),
        pos("token", U64, ""),
    ],
};

/// A timed step.
pub static TIME_STEP: Kind = Kind {
    name: "time_step",
    head: "time step",
    doc: "Generated step i's wall; warm marks a step --warm drops from the statistics.",
    parts: &[
        lit(" "),
        pos("i", U64, ""),
        flag("warm"),
        key("ms", F64(4), "ms"),
    ],
};

/// A timed draft pass.
pub static TIME_PASS: Kind = Kind {
    name: "time_pass",
    head: "time pass",
    doc: "Draft pass i's wall, the positions it advanced and what it ran.",
    parts: &[
        lit(" "),
        pos("i", U64, ""),
        flag("warm"),
        key("ms", F64(4), "ms"),
        key("positions", U64, "positions"),
        key("kind", Word, ""),
    ],
};

/// The generated tokens.
pub static TOKENS: Kind = Kind {
    name: "tokens",
    head: "tokens",
    doc: "The generated tokens, token 0 first.",
    parts: &[lit(" "), pos("tokens", List, "")],
};

/// One generated step's host-tier counters.
pub static STAT_STEP: Kind = Kind {
    name: "stat_step",
    head: "stat step",
    doc: "BLOOMERY_STEP_STATS: one generated step's host-tier counters, page faults, free device bytes and engram rows; the step's go waits (the card's time from the host's signal to the next go, one per service: least, lower median, most) and the host time in its synchronous parameter copy.",
    parts: &[
        lit(" "),
        pos("i", U64, ""),
        flag("warm"),
        key("served", U64, ""),
        key("leg_us", F64(1), "us"),
        key("straggle_us", F64(1), "us"),
        key("straggle_max_us", F64(1), "us"),
        key("host_slots", U64, ""),
        key("host_w2", F64(4), ""),
        key("overlap", U64, ""),
        key("union", U64, ""),
        key("go_early", U64, ""),
        key("parks", U64, ""),
        key("majflt", U64, ""),
        key("minflt", U64, ""),
        key("vram_free", U64, "B"),
        key("eng_warm", U64, ""),
        key("eng_cold", U64, ""),
        key("eng_direct", U64, ""),
        key("eng_wait_us", F64(1), "us"),
        key("eng_helper_us", F64(1), "us"),
        key("eng_classify_us", F64(1), "us"),
        key("gap_min_us", F64(1), "us"),
        key("gap_p50_us", F64(1), "us"),
        key("gap_max_us", F64(1), "us"),
        key("params_us", F64(1), "us"),
    ],
};

/// The steps' summary.
pub static STAT_SUMMARY: Kind = Kind {
    name: "stat_summary",
    head: "stat summary",
    doc: "BLOOMERY_STEP_STATS: the kept steps' host-tier, fault, device-memory and engram statistics.",
    parts: &[
        key("steps", U64, ""),
        key("leg_us_mean", F64(1), "us"),
        key("leg_us_p50", F64(1), "us"),
        key("straggle_us_max", F64(1), "us"),
        key("host_slots_mean", F64(1), ""),
        key("phi_mean", F64(4), ""),
        key("majflt", U64, ""),
        key("minflt", U64, ""),
        key("vram_free_load", U64, "B"),
        key("vram_free_min", U64, "B"),
        key("eng_helper", Word, ""),
        key("eng_warm", U64, ""),
        key("eng_cold", U64, ""),
        key("eng_direct", U64, ""),
        key("eng_wait_us_mean", F64(1), "us"),
        key("eng_wait_us_p50", F64(1), "us"),
        key("eng_wait_us_max", F64(1), "us"),
        key("eng_helper_us_mean", F64(1), "us"),
        key("eng_classify_us_mean", F64(1), "us"),
    ],
};

/// One generated step's host-tier counters on a body with a host tier and
/// no engram rows.
pub static STAT_STEP_HOST: Kind = Kind {
    name: "stat_step_host",
    head: "stat step",
    doc: "BLOOMERY_STEP_STATS: one generated step's host-tier counters, page faults and free device bytes (a body with a host tier and no engram rows: generate_qwen3moe, generate_glm5next). `gap_min_us`, `gap_p50_us` and `gap_max_us` are the step's go waits, one per service; a step no service ran in prints zeros.",
    parts: &[
        lit(" "),
        pos("i", U64, ""),
        flag("warm"),
        key("served", U64, ""),
        key("leg_us", F64(1), "us"),
        key("straggle_us", F64(1), "us"),
        key("straggle_max_us", F64(1), "us"),
        key("host_slots", U64, ""),
        key("host_w2", F64(4), ""),
        key("go_early", U64, ""),
        key("parks", U64, ""),
        key("majflt", U64, ""),
        key("minflt", U64, ""),
        key("vram_free", U64, "B"),
        key("gap_min_us", F64(1), "us"),
        key("gap_p50_us", F64(1), "us"),
        key("gap_max_us", F64(1), "us"),
    ],
};

/// The steps' summary on a body with a host tier and no engram rows.
pub static STAT_SUMMARY_HOST: Kind = Kind {
    name: "stat_summary_host",
    head: "stat summary",
    doc: "BLOOMERY_STEP_STATS: the kept steps' host-tier, fault and device-memory statistics (a body with a host tier and no engram rows: generate_qwen3moe, generate_glm5next).",
    parts: &[
        key("steps", U64, ""),
        key("leg_us_mean", F64(1), "us"),
        key("leg_us_p50", F64(1), "us"),
        key("straggle_us_max", F64(1), "us"),
        key("host_slots_mean", F64(1), ""),
        key("majflt", U64, ""),
        key("minflt", U64, ""),
        key("vram_free_load", U64, "B"),
        key("vram_free_min", U64, "B"),
    ],
};

/// A generated step through the finite probe.
pub static STAT_FINITE_STEP: Kind = Kind {
    name: "stat_finite_step",
    head: "stat finite step",
    doc: "BLOOMERY_CHECK_FINITE: generated step k's position and the probe's reading of it.",
    parts: &[
        lit(" "),
        pos("step", U64, ""),
        lit(" pos "),
        pos("pos", U64, ""),
        lit(" "),
        pos("observed", Text, ""),
    ],
};

/// A fed position the probe found wanting.
pub static STAT_FINITE_FED: Kind = Kind {
    name: "stat_finite_fed",
    head: "stat finite fed",
    doc: "BLOOMERY_CHECK_FINITE: a fed position that is not ok, and the probe's reading of it.",
    parts: &[
        lit(" pos "),
        pos("pos", U64, ""),
        lit(" "),
        pos("observed", Text, ""),
    ],
};

/// The finite probe's summary.
pub static STAT_FINITE_SUMMARY: Kind = Kind {
    name: "stat_finite_summary",
    head: "stat finite summary",
    doc: "BLOOMERY_CHECK_FINITE: the positions probed, those with a non-finite seam, the first of them, and the eager argmaxes that differ.",
    parts: &[
        key("positions", U64, ""),
        key("nonfinite_positions", U64, ""),
        key("first", Text, ""),
        key("eager_differs", U64, ""),
    ],
};

/// A draft's summary.
pub static DRAFT_SUMMARY: Kind = Kind {
    name: "draft_summary",
    head: "draft summary",
    doc: "BLOOMERY_DRAFT: proposals, accepts, positions and passes over the run, the kept passes' \
          positions per second, the draft, and the width chooser's mode (`BLOOMERY_MTP_WIDTH`, \
          cost or fixed); under cost the chooser's state at the run's end (the gate's width, 0 \
          closed; each width's cost median, index 0 the plain step, `-` while unread; the mean \
          acceptance profile it rates them by) and the passes it ran closed.",
    parts: &[
        key("proposals", U64, ""),
        key("accepts", U64, ""),
        key("positions", U64, "positions"),
        key("passes", U64, ""),
        key("tok/s(positions)", F64(2), "tok/s"),
        key("kind", Word, ""),
        key("width", Word, ""),
        opt("gate", U64, ""),
        opt("costs", Csv, "ms"),
        opt("a", Csv, ""),
        opt("closed", U64, "passes"),
    ],
};

/// The MTP draft's join to a held sequence a prompt call continues.
pub static MTP_PROMPT: Kind = Kind {
    name: "mtp_prompt",
    head: "mtp prompt",
    doc: "BLOOMERY_DRAFT=mtp: a prompt call that continues the held sequence: its first \
          position, the rows the draft walked to catch its store up to it, and why the draft \
          proposes nothing for the call's generation (none when it drafts).",
    parts: &[
        key("start", U64, "positions"),
        key("caught_up", U64, ""),
        key("skipped", Text, ""),
    ],
};

/// The MTP draft's window summary.
pub static MTP_SUMMARY: Kind = Kind {
    name: "mtp_summary",
    head: "mtp summary",
    doc: "BLOOMERY_DRAFT=mtp: the windows' proposals and the rows each kept (a count a kept \
          length, 1 to the window's rows: 4 for Qwen3.8, 2 for GLM-5.3), every pass by the ids \
          it verified (one count a width, 0 a plain step; the draft's whole width on every pass \
          under `BLOOMERY_MTP_WIDTH=fixed`), the positions and passes over the run, and the \
          kept passes' positions per second.",
    parts: &[
        key("proposals", U64, ""),
        key("kept", List, "windows"),
        key("widths", List, ""),
        key("positions", U64, "positions"),
        key("passes", U64, ""),
        key("tok/s(positions)", F64(2), "tok/s"),
    ],
};

/// One drafted window of Qwen3.8's MTP draft.
pub static MTP_WINDOW: Kind = Kind {
    name: "mtp_window",
    head: "mtp window",
    doc: "BLOOMERY_MTP_WINDOWS: one drafted window, after the arm's mtp summary: its pass (the \
          time pass row's number), the target's position before its verify, the proposal's ids, \
          each one's probability among the draft head's rows (as the chain read it back, in the \
          proposal's order), how many of those ids the pass verified (the width chooser's \
          cut; the whole proposal under `BLOOMERY_MTP_WIDTH=fixed`) and how many of them the \
          target kept.",
    parts: &[
        key("window", U64, ""),
        key("pos", U64, "positions"),
        key("ids", Csv, ""),
        key("p", Csv, ""),
        key("width", U64, ""),
        key("accepted", U64, ""),
    ],
};

/// A drafting server's width chooser over one request.
pub static MTP_WIDTH: Kind = Kind {
    name: "mtp_width",
    head: "mtp width",
    doc: "BLOOMERY_MTP_WIDTH=cost: the width chooser's passes over one request of a drafting \
          server, printed at the slot's next prompt call: the passes that verified a proposal \
          (windows), the windows by the rows they kept (one count a kept length, 1 to the \
          draft's width + 1), every pass by the ids it verified (one count a width, 0 the plain \
          step: the gate closed, the warm-up's or a probe's plain turn, a shadow's pass), E, the \
          mean rows a window kept; then the chooser's state at the request's end (the gate's \
          width, 0 closed; each width's cost median, index 0 the plain step, `-` while unread; \
          the mean acceptance profile it rates them by) and the request's passes of its own the \
          gate stood closed for (a round of several slots' windows is none).",
    parts: &[
        key("windows", U64, ""),
        key("kept", Csv, "windows"),
        key("widths", Csv, "passes"),
        key("e", F64(3), ""),
        key("gate", U64, ""),
        key("costs", Csv, "ms"),
        key("a", Csv, ""),
        key("closed", U64, "passes"),
    ],
};

/// The timed run's footer.
pub static SMOKE: Kind = Kind {
    name: "smoke",
    head: "SMOKE",
    doc: "--time's footer: the kept steps' p50 and mean and the rate at the p50; under a draft the positions and their rate too.",
    parts: &[
        key("mode", Word, ""),
        key("place", Word, ""),
        key("prompt_tokens", U64, "positions"),
        key("depth", U64, "positions"),
        key("generated", U64, ""),
        key("warm", U64, ""),
        key("steps", U64, ""),
        key("p50_ms", F64(4), "ms"),
        key("mean_ms", F64(4), "ms"),
        key("tok/s(p50)", F64(2), "tok/s"),
        opt("positions", U64, "positions"),
        opt("tok/s(positions)", F64(2), "tok/s"),
    ],
};

/// The chat prompt's ids.
pub static PROMPT_IDS: Kind = Kind {
    name: "prompt_ids",
    head: "prompt_ids",
    doc: "The prompt's ids as the tokenizer encoded it.",
    parts: &[lit(" "), pos("ids", List, "")],
};

/// The chat's drawn ids.
pub static IDS: Kind = Kind {
    name: "ids",
    head: "ids",
    doc: "The ids the chat drew, the stop's included.",
    parts: &[lit(" "), pos("ids", List, "")],
};

/// Whether the chat's text is its ids' decode.
pub static TEXT_CONSISTENT: Kind = Kind {
    name: "text_consistent",
    head: "text_consistent=",
    doc: "Whether the text streamed piece by piece is the drawn ids' whole decode.",
    parts: &[pos("consistent", Bool, "")],
};

/// The chat's end.
pub static CHAT: Kind = Kind {
    name: "chat",
    head: "chat:",
    doc: "The chat's prompt length, the ids drawn and why it stopped.",
    parts: &[
        key("prompt_tokens", U64, "positions"),
        key("generated", U64, ""),
        key("stop", Word, ""),
    ],
};

/// The server's address.
pub static LISTENING: Kind = Kind {
    name: "listening",
    head: "bloomery-serve-ds41:",
    doc: "The server's placement, context and the address it listens on.",
    parts: &[
        key("place", Word, ""),
        key("ctx", U64, "positions"),
        lit(" listening on http://"),
        pos("addr", Word, ""),
    ],
};

/// The Qwen3.8 server's address.
pub static LISTENING38: Kind = Kind {
    name: "listening38",
    head: "bloomery-serve-qwen38:",
    doc: "The Qwen3.8 server's placement, the context a slot serves (`ctx`, the rows each slot's caches hold; `slot_ctx` the same number, a resident slot's share of the split context), the resident sequences it serves and the address it listens on.",
    parts: &[
        key("place", Word, ""),
        key("ctx", U64, "positions"),
        key("slots", U64, ""),
        key("slot_ctx", U64, "positions"),
        lit(" listening on http://"),
        pos("addr", Word, ""),
    ],
};

/// The GLM server's address.
pub static LISTENING_GLM: Kind = Kind {
    name: "listening_glm",
    head: "bloomery-serve-glm:",
    doc: "The GLM server's placement, context and the address it listens on.",
    parts: &[
        key("place", Word, ""),
        key("ctx", U64, "positions"),
        lit(" listening on http://"),
        pos("addr", Word, ""),
    ],
};

/// The Qwen3 seat's address.
pub static LISTENING_QWEN3: Kind = Kind {
    name: "listening_qwen3",
    head: "bloomery-serve-qwen3:",
    doc: "The Qwen3 seat's architecture, the context a slot serves (`ctx`, the cache rows the model was loaded with; `slot_ctx` the same number — a resident slot's share of the split context, or the whole context the slots take in turns), the slots it serves and the address it listens on.",
    parts: &[
        key("arch", Word, ""),
        key("ctx", U64, "positions"),
        key("slots", U64, ""),
        key("slot_ctx", U64, "positions"),
        lit(" listening on http://"),
        pos("addr", Word, ""),
    ],
};

/// The decide seat's address.
pub static LISTENING_DECIDE: Kind = Kind {
    name: "listening_decide",
    head: "bloomery-serve-decide:",
    doc: "The decide seat's model, quant, head, row and the address it listens on.",
    parts: &[
        key("model", Word, ""),
        key("quant", Text, ""),
        key("head", Word, ""),
        key("row", Word, ""),
        lit(" listening on http://"),
        pos("addr", Word, ""),
    ],
};

/// The Qwen3 seat's load.
pub static LOAD_QWEN3: Kind = Kind {
    name: "load_qwen3",
    head: "load",
    doc: "The Qwen3 seat's model on device 0: its architecture, resident bytes (every resident sequence's), the K/V planes' format, the cache rows the model was loaded with (`ctx`; `slot_ctx` the same number, each resident sequence's), the resident sequences it holds (`slots`; 1 when the server's slots take one sequence in turns), layers, the prompt ubatch and the step graph's nodes, and the load's wall.",
    parts: &[
        key("arch", Word, ""),
        key("resident_bytes", U64, "B"),
        key("cache", Word, ""),
        key("ctx", U64, "positions"),
        key("slots", U64, ""),
        key("slot_ctx", U64, "positions"),
        key("layers", U64, ""),
        key("ubatch", U64, "positions"),
        key("graph_nodes", U64, ""),
        lit(" in "),
        pos("load_s", F64(1), "s"),
        lit(" s"),
    ],
};

/// The GGUF set a `--hf` quant picked.
pub static HF_SET: Kind = Kind {
    name: "hf_set",
    head: "hf set",
    doc: "The GGUF set `--hf <repo>[:<quant>]` picked in the repo's listing: its name, its tag, its files and their bytes.",
    parts: &[
        key("repo", Word, ""),
        key("set", Word, ""),
        key("tag", Word, ""),
        key("files", U64, ""),
        key("bytes", U64, "B"),
    ],
};

/// A file of a `--hf` fetch before it runs.
pub static HF_FILE: Kind = Kind {
    name: "hf_file",
    head: "hf file",
    doc: "A file of a `--hf` fetch: its path in the repo, its bytes, what the cache held (cached: checked, nothing fetched; fetch: from byte 0; resume: a partial file fetched from its end; stale: a cached file whose content is not the listing's, fetched again; offline-cached: a verified file used with no listing, after an `hf offline` line) and the byte the fetch starts at.",
    parts: &[
        key("file", Word, ""),
        key("bytes", U64, "B"),
        key("state", Word, ""),
        key("from", U64, "B"),
    ],
};

/// A `--hf` start with no listing, the cache standing in.
pub static HF_OFFLINE: Kind = Kind {
    name: "hf_offline",
    head: "hf offline",
    doc: "A `--hf` start with no listing — it failed at the network (no host, no connection, a timeout, no TLS, no reply), the hub answered busy or down (HTTP 429 or 5xx), or HF_HUB_OFFLINE asked for none: the cache's one verified set the quant names is used, its files `offline-cached`, and the reason.",
    parts: &[key("repo", Word, ""), key("reason", Text, "")],
};

/// A `--hf` repo's MTP draft left unfetched.
pub static HF_DRAFT_SKIP: Kind = Kind {
    name: "hf_draft_skip",
    head: "hf draft skipped",
    doc: "The `--hf` repo's MTP draft file, not fetched: the set picked cannot run it, and why, as the plan's own rule (`PlanInputs::mtp_borrows`) says it.",
    parts: &[key("file", Word, ""), key("reason", Text, "")],
};

/// A running `--hf` download.
pub static HF_PROGRESS: Kind = Kind {
    name: "hf_progress",
    head: "hf progress",
    doc: "A running `--hf` download's bytes on disk, every few seconds.",
    parts: &[
        key("file", Word, ""),
        key("have", U64, "B"),
        key("of", U64, "B"),
    ],
};

/// A `--hf` file checked.
pub static HF_DONE: Kind = Kind {
    name: "hf_done",
    head: "hf done",
    doc: "A `--hf` file checked against the listing: the bytes this run fetched, the check (the LFS sha256, or the git blob sha1 of a file stored in git) and its digest.",
    parts: &[
        key("file", Word, ""),
        key("fetched_bytes", U64, "B"),
        key("check", Word, ""),
        key("digest", Word, ""),
    ],
};

/// The GLM seat's NextN draft, loaded beside the target.
pub static LOAD_DRAFT_GLM: Kind = Kind {
    name: "load_draft_glm",
    head: "load draft=mtp",
    doc: "The GLM seat's MTP draft, the file's NextN layer loaded beside the target, after the \
          load line: the layer's index in the file, its weights' and store's device bytes, its \
          walk's arena, the head it scores with, the card bytes the plan reserved for it, and the \
          load's wall.",
    parts: &[
        key("layer", U64, ""),
        key("resident", U64, "B"),
        key("arena", U64, "B"),
        key("head", Word, ""),
        key("plan_bytes", U64, "B"),
        lit(" in "),
        pos("load_s", F64(1), "s"),
        lit(" s (runtime value)"),
    ],
};

/// A GLM seat that drafts nothing ([`bloomery_levers::GlmWhy`]).
pub static LOAD_DRAFT_OFF_GLM: Kind = Kind {
    name: "load_draft_off_glm",
    head: "load draft=off",
    doc: "A GLM seat that drafts nothing, after its load line, and why: BLOOMERY_DRAFT=off, or \
          unset and the condition that left the plain path (the draft unset record's).",
    parts: &[lit(" ("), pos("why", Text, ""), lit(")")],
};

/// The placement a serving seat runs by: `--place` set, or unset and the
/// common rule's choice (`generate::Place::choose`).
pub static PLACE_UNSET: Kind = Kind {
    name: "place_unset",
    head: "place unset",
    doc: "The placement a serving seat runs by, after the levers' records and before the ctx line \
          and the plan, and why: set (--place, its word as the plan line prints it), or unset and \
          the common rule's choice on the cards — one card, the offer itself (one card); two \
          cards, a when the body serves no tier card (two cards, the body serves no tier card) or \
          no sitting has shown the tier not slower (two cards, no sitting has shown the tier not \
          slower), and else by the plan of bp (the largest the stage, the next-largest its expert \
          tier): kept when its tier holds at least the break-even and more than no expert (two \
          cards, tier at or past the break-even), else a (two cards, tier under the break-even; \
          two cards, the tier holds no expert; two cards, the tier's plan refused, the refusal on \
          stderr). tier_experts is the experts the tier cards hold in that plan of bp, only where \
          the rule asked the plan for them (never under --place, on one card, or on a refused \
          plan); break_even and basis name the family's rule, basis the card file that decides it, \
          whenever the family has them.",
    parts: &[
        key("place", Word, ""),
        key("why", Text, ""),
        opt("tier_experts", U64, "experts"),
        opt("break_even", U64, "experts"),
        opt("basis", Word, ""),
    ],
};

/// What `BLOOMERY_DRAFT` unset resolved to on the GLM seat
/// ([`bloomery_levers::glm_unset`]).
pub static DRAFT_UNSET_GLM: Kind = Kind {
    name: "draft_unset_glm",
    head: "draft unset",
    doc: "What BLOOMERY_DRAFT unset resolved to on the GLM seat before the plan: the word the load \
          runs by (mtp, the file's NextN layer, or off) and why (--place a, or what left the plain \
          path: --place gate, a file of other than one next-token layer, stores short of a window).",
    parts: &[key("draft", Word, ""), key("why", Text, "")],
};

/// An unset MTP draft yielding to the context
/// ([`bloomery_levers::DraftYield`]): the card bytes it reserves at the
/// context it denied, the positions a slot gets with it and without it,
/// and the base the ctx rule aims for. The seats print it where the
/// decision closed, before the `ctx` line it produced; only a load that
/// yields prints one.
pub static DRAFT_YIELD: Kind = Kind {
    name: "draft_yield",
    head: "draft yield",
    doc: "An unset MTP draft yielding to the context, before the ctx line of the plain rule it \
          fell to: the draft's card bytes at the context it denied, the positions a slot gets \
          with the draft and without it, and the base the ctx rule aims for.",
    parts: &[
        key("card_bytes", U64, "B"),
        key("with", U64, "positions"),
        key("without", U64, "positions"),
        key("base", U64, "positions"),
    ],
};

/// What `BLOOMERY_RESIDENCY` unset resolved to on the GLM seat
/// ([`bloomery_levers::glm_unset`]).
pub static RESIDENCY_UNSET_GLM: Kind = Kind {
    name: "residency_unset_glm",
    head: "residency unset",
    doc: "What BLOOMERY_RESIDENCY unset resolved to on the GLM seat, after the plan line: the word \
          the load runs by and why (--place a, or what left it off: --place gate, stores short of \
          a window, the step feed, or the plan: no card expert, no room, or the churn pool past \
          the plan's host headroom or past what MemAvailable leaves).",
    parts: &[key("residency", Word, ""), key("why", Text, "")],
};

/// A round of several slots the seat served, under `BLOOMERY_STEP_STATS`.
pub static SLOTS_ROUND: Kind = Kind {
    name: "slots round",
    head: "slots round",
    doc: "BLOOMERY_STEP_STATS: one round of several slots the seat served — its \
          command (step: a step a row; pass: a drafted pass a row), the round's \
          rows, the passes the seat ran them as (the fallback loop's rows, one \
          a row; a one-pass seat's 1, or its cuts at its body's pass bound), and \
          the resident sequences the seat serves.",
    parts: &[
        key("cmd", Word, ""),
        key("rows", U64, ""),
        key("passes", U64, ""),
        key("slots", U64, ""),
    ],
};

/// The server's prompt cache: its budget, the host headroom it was derived
/// from, and the token a prompt call is cut at.
pub static CACHE_CONFIG: Kind = Kind {
    name: "cache_config",
    head: "cache",
    doc: "The prompt cache's budget, the plan's host headroom, and the user-message token prompt calls are cut at, with whether the chat template writes it.",
    parts: &[
        key("ram", U64, "B"),
        key("headroom", I64, "B"),
        key("user_start", Word, ""),
        key("in_template", Bool, ""),
    ],
};

/// A request whose shared prefix the engine keeps less of.
pub static CACHE_REUSE: Kind = Kind {
    name: "cache_reuse",
    head: "cache reuse",
    doc: "A request shared common ids with the slot, asked to keep ask of them, and the engine kept fewer, for the reason given.",
    parts: &[
        key("common", U64, "positions"),
        key("ask", U64, "positions"),
        key("kept", U64, "positions"),
        key("held", U64, "positions"),
        key("reason", Text, ""),
    ],
};

/// The slot's state into the prompt cache.
pub static CACHE_SAVE: Kind = Kind {
    name: "cache_save",
    head: "cache save",
    doc: "The slot's state went into the prompt cache: its positions and bytes, the server's wall clock around the snapshot, and the cache after it.",
    parts: &[
        key("positions", U64, "positions"),
        key("bytes", U64, "B"),
        key("ms", F64(3), "ms"),
        key("entries", U64, ""),
        key("cache_bytes", U64, "B"),
    ],
};

/// A cached state into the slot.
pub static CACHE_LOAD: Kind = Kind {
    name: "cache_load",
    head: "cache load",
    doc: "A cached state replaced the slot's: its positions, the ids it shares with the request, what the engine keeps of them after it and what the slot would have kept, its bytes and the server's wall clock around the resume.",
    parts: &[
        key("positions", U64, "positions"),
        key("common", U64, "positions"),
        key("kept", U64, "positions"),
        key("slot_kept", U64, "positions"),
        key("bytes", U64, "B"),
        key("ms", F64(3), "ms"),
    ],
};

/// A state out of the prompt cache.
pub static CACHE_EVICT: Kind = Kind {
    name: "cache_evict",
    head: "cache evict",
    doc: "A cached state left the cache: its positions, its bytes, and why.",
    parts: &[
        key("positions", U64, "positions"),
        key("bytes", U64, "B"),
        key("why", Text, ""),
    ],
};

/// A state the prompt cache did not keep.
pub static CACHE_SKIP: Kind = Kind {
    name: "cache_skip",
    head: "cache skip",
    doc: "The slot's state was not cached, or a cached state was not taken back, and why.",
    parts: &[key("positions", U64, "positions"), key("why", Text, "")],
};

/// A prompt call cut at message starts.
pub static PREFILL_SPLIT: Kind = Kind {
    name: "prefill_split",
    head: "prefill split",
    doc: "A prompt call ran as calls cut at the positions given, so a later request keeps them.",
    parts: &[
        key("first", U64, "positions"),
        key("end", U64, "positions"),
        key("at", Csv, "positions"),
    ],
};

/// One sequence of a layer-tap dump written.
pub static TAPS_SEQ: Kind = Kind {
    name: "taps_seq",
    head: "taps seq",
    doc: "A tap dump's sequence, its files written and their sizes checked: where its prompt was cut \
          from, its prompt and total positions, its bytes, and the host wall it took (a runtime \
          value).",
    parts: &[
        key("k", U64, ""),
        key("source", Word, ""),
        key("offset", U64, "positions"),
        key("n_prompt", U64, "positions"),
        key("n_total", U64, "positions"),
        key("bytes", U64, "B"),
        key("wall_s", F64(1), "s"),
    ],
};

/// A layer-tap dump finished.
pub static TAPS_DUMP: Kind = Kind {
    name: "taps_dump",
    head: "taps dump",
    doc: "A tap dump's directory, its sequences, positions and bytes as its manifest reads back, the \
          layers each position's row holds, the path its prompts ran on, and the host wall (a \
          runtime value).",
    parts: &[
        key("dir", Word, ""),
        key("seqs", U64, ""),
        key("positions", U64, "positions"),
        key("bytes", U64, "B"),
        key("layers", Csv, ""),
        key("prefill", Word, ""),
        key("wall_s", F64(1), "s"),
    ],
};

/// What `generate_ds41` prints, in the order it prints them.
pub static GENERATE_DS41: &[&Kind] = &[
    &PLACE_UNSET,
    &PLAN,
    &CALL_PLAN,
    &CALL_GROUPS,
    &CALL_BATCH,
    &CALL_LAYER,
    &CALL_NEED,
    &CALL_LB,
    &LOAD,
    &LOAD_PHASES,
    &HOST_POPULATE,
    &HOST_POPULATE_OFF,
    &HOST_LOCK,
    &HELPER,
    &LOAD_DRAFT,
    &CAPTURE,
    &CAPTURE_PAIR,
    &PREFILL_BYTES,
    &CHECK_FINITE,
    &ARM,
    &FED,
    &STEP0,
    &STAT_PREFILL,
    &STAT_PREFILL_CED,
    &STAT_PREFILL_SPLIT,
    &STAT_PREFILL_FRONT,
    &STAT_PREFILL_LB,
    &TIME_PROMPT,
    &STEP,
    &TIME_STEP,
    &TIME_PASS,
    &TOKENS,
    &LOGITS,
    &TOP2,
    &ROWS,
    &IGNORE_EOS,
    &SLOT_TABLE,
    &TABLE_DUMP,
    &CARD_TABLE,
    &STAT_STEP,
    &STAT_SUMMARY,
    &STAT_FINITE_STEP,
    &STAT_FINITE_FED,
    &STAT_FINITE_SUMMARY,
    &DRAFT_SUMMARY,
    &SMOKE,
    &RESIDENCY_LEVER,
    &RESIDENCY_HOST,
    &RESIDENCY_PASS,
    &RESIDENCY_RESET,
    &RESIDENCY_LEAK,
    &CALL_STREAM,
    &CALL_STREAM_END,
];

/// What `bloomery-chat` prints, all on stderr.
pub static BLOOMERY_CHAT: &[&Kind] = &[
    &PLACE_UNSET,
    &PROMPT_IDS,
    &PLAN,
    &LOAD_GENERATOR,
    &LOAD_PHASES,
    &HOST_POPULATE,
    &HOST_POPULATE_OFF,
    &HOST_LOCK,
    &CAPTURE,
    &IDS,
    &TEXT_CONSISTENT,
    &CHAT,
];

/// What `bloomery-serve-ds41` prints, all on stderr, and the rounds of
/// several slots a `BLOOMERY_STEP_STATS` run counts.
pub static BLOOMERY_SERVE_DS41: &[&Kind] = &[
    &PLACE_UNSET,
    &PLAN,
    &CACHE_CONFIG,
    &LOAD_GENERATOR,
    &LOAD_PHASES,
    &HOST_POPULATE,
    &HOST_POPULATE_OFF,
    &HOST_LOCK,
    &HELPER,
    &LOAD_DRAFT,
    &CAPTURE,
    &CAPTURE_PAIR,
    &LISTENING,
    &CACHE_REUSE,
    &CACHE_SAVE,
    &CACHE_LOAD,
    &CACHE_EVICT,
    &CACHE_SKIP,
    &PREFILL_SPLIT,
    &RESIDENCY_LEVER,
    &RESIDENCY_HOST,
    &RESIDENCY_PASS,
    &RESIDENCY_RESET,
    &RESIDENCY_LEAK,
    &MTP_WIDTH,
    &SLOTS_ROUND,
];

/// What `bloomery-serve-qwen38` prints, all on stderr.
pub static BLOOMERY_SERVE_QWEN38: &[&Kind] = &[
    &PLAN38,
    &LISTENING38,
    &CACHE_REUSE,
    &MTP_PROMPT,
    &LOAD_DRAFT_OFF38,
    &MTP_HEAD38,
    &MTP_KEEP38,
    &RESIDENCY_LEVER,
    &PLACE_UNSET,
    &RESIDENCY_UNSET,
    &RESIDENCY_HOST,
    &CALL_STREAM,
    &CALL_STREAM_END,
    &RESIDENCY_PASS,
    &RESIDENCY_RESET,
    &SLOTS_ROUND,
    &MTP_WIDTH,
    &DRAFT_YIELD,
    &NVTIER,
    &TIER_CENSUS,
];

/// What the Qwen3 seat of `bloomery-serve` prints, all on stderr: the
/// placement and why (`place unset`), a `--hf`
/// fetch's lines (printed by the server before the seat registers), the
/// load, the address and each request's note of the prefix it kept none of.
pub static BLOOMERY_SERVE_QWEN3: &[&Kind] = &[
    &PLACE_UNSET,
    &PLAN38,
    &HF_OFFLINE,
    &HF_SET,
    &HF_FILE,
    &HF_PROGRESS,
    &HF_DONE,
    &LOAD_QWEN3,
    &LISTENING_QWEN3,
    &CACHE_REUSE,
    &SLOTS_ROUND,
];

/// What the GLM seat of `bloomery-serve` prints, all on stderr: the
/// residency lever's word, the placement and why, the `plan`, `load` and
/// `capture` lines `generate_glm5next` prints, the host set's records of a
/// placed load, the draft's load line and its verify's capture, the
/// listening line, the reuse records the checkpoint rule answers, the
/// draft's joins, the residency machine's records, and the rounds of several
/// slots a `BLOOMERY_STEP_STATS` run counts.
pub static BLOOMERY_SERVE_GLM: &[&Kind] = &[
    &RESIDENCY_LEVER,
    &DRAFT_UNSET_GLM,
    &DRAFT_YIELD,
    &PLACE_UNSET,
    &PLAN,
    &RESIDENCY_UNSET_GLM,
    &RESIDENCY_HOST,
    &LOAD_GENERATOR,
    &HOST_POPULATE,
    &HOST_POPULATE_OFF,
    &HOST_LOCK,
    &LOAD_DRAFT_GLM,
    &LOAD_DRAFT_OFF_GLM,
    &CAPTURE,
    &CAPTURE_PAIR,
    &LISTENING_GLM,
    &CACHE_REUSE,
    &MTP_PROMPT,
    &RESIDENCY_PASS,
    &RESIDENCY_RESET,
    &RESIDENCY_LEAK,
    &SLOTS_ROUND,
    &MTP_WIDTH,
];

/// What `generate_glm5next` prints, in the order it prints them.
pub static GENERATE_GLM5NEXT: &[&Kind] = &[
    &RESIDENCY_LEVER,
    &PLACE_UNSET,
    &PLAN,
    &RESIDENCY_HOST,
    &LOAD_GENERATOR,
    &LOAD_PHASES,
    &HOST_POPULATE,
    &HOST_POPULATE_OFF,
    &HOST_LOCK,
    &CAPTURE,
    &CAPTURE_PAIR,
    &PROMPT_UNITS,
    &ARM,
    &FED,
    &STEP0,
    &STAT_PROMPT_LB,
    &STAT_PROMPT_SPLIT,
    &TIME_PROMPT,
    &STEP,
    &TIME_STEP,
    &TIME_PASS,
    &TOKENS,
    &LOGITS,
    &TOP2,
    &ROWS,
    &IGNORE_EOS,
    &SLOT_TABLE,
    &TABLE_DUMP,
    &CARD_TABLE,
    &MTP_SUMMARY,
    &SMOKE,
    &STAT_STEP_HOST,
    &STAT_SUMMARY_HOST,
    &RESIDENCY_PASS,
    &RESIDENCY_RESET,
    &RESIDENCY_LEAK,
];

/// What `generate_qwen3moe` prints as records: a load's `plan` (a qwen4exp
/// line; a placed qwen3moe or qwen35moe line names its architecture), the
/// common unset rule's `place unset` (after a set residency lever's word,
/// before the load), its residency lever's word (set, before the load;
/// unset, after the `plan`, or before the load on another family and under
/// `--dump-taps`)
/// and, drafting nothing, why after the `load` line, drafting, its head
/// after the `load draft=mtp` line; under `--dump-taps`,
/// after its `load` line; under `--time`, a drafted arm's passes and a
/// `BLOOMERY_GEN_SLOTS` arm's rounds as `time pass` records; a drafted arm's
/// `mtp summary` and, under `BLOOMERY_MTP_WINDOWS`, its `mtp window` records
/// after its lines; and under `BLOOMERY_STEP_STATS`, after a qwen4exp run's
/// lines; the binary's other lines are its own.
pub static GENERATE_QWEN3MOE: &[&Kind] = &[
    &PLAN38,
    &LOAD_DRAFT_OFF38,
    &MTP_HEAD38,
    &TAPS_SEQ,
    &TAPS_DUMP,
    &TIME_PASS,
    &MTP_SUMMARY,
    &MTP_WINDOW,
    &STAT_STEP_HOST,
    &STAT_SUMMARY_HOST,
    &STAT_PROMPT_SPLIT,
    &STAT_PROMPT_LB,
    &RESIDENCY_LEVER,
    &PLACE_UNSET,
    &RESIDENCY_UNSET,
    &RESIDENCY_HOST,
    &RESIDENCY_PASS,
    &RESIDENCY_RESET,
    &CALL_STREAM,
    &CALL_STREAM_END,
    &XSTREAM,
    &XSTREAM_END,
    &NVTIER,
];

/// The records `clef_hidden` prints: the prompt call's, then, given row ids,
/// the output rows'.
pub static CLEF_HIDDEN_BIN: &[&Kind] = &[&CLEF_HIDDEN, &CLEF_ROWS];

/// The record `gate_deepseek41_prefill` prints inside its own lines, after
/// its name and the case's.
pub static GATE_DEEPSEEK41_PREFILL: &[&Kind] = &[&STAT_PREFILL_SPLIT];

/// The records `probe_nvread` prints: one row an arm and case.
pub static PROBE_NVREAD: &[&Kind] = &[&NVREAD];

/// The width chooser's state on `r`, a `draft summary` or an `mtp width`
/// record: the gate's width, each width's cost median in ms (`-` while
/// unread), the mean acceptance profile, and `closed`, its own passes it
/// ran closed over what the record covers (`runtime::width::Tally::closed`).
#[cfg(feature = "gpu")]
pub fn width_gate(r: Record, s: &runtime::width::GateState, closed: u64) -> Record {
    let costs = s.costs.iter().map(|c| match c {
        Some(ms) => format!("{ms:.2}"),
        None => "-".to_string(),
    });
    r.u("gate", s.width)
        .csv("costs", costs)
        .csv("a", s.a.iter().map(|a| format!("{a:.3}")))
        .u("closed", closed)
}

/// `gate_nvtier`'s rows: the tier's counters, the residency rule's two
/// records of the paged plan and the tier census of each phase.
pub static GATE_NVTIER: &[&Kind] = &[&NVTIER, &RESIDENCY_UNSET, &RESIDENCY_HOST, &TIER_CENSUS];

/// The `plan` record of `plan`, made over `machine` by the placement named
/// `place`: its first card, the experts on it and on the host, the per-layer
/// card counts' range, and the budget.
pub fn plan(place: &str, machine: &Machine, plan: &Plan<'_>) -> Record {
    let held: Vec<u64> = plan.n_l.iter().copied().filter(|&n| n > 0).collect();
    let card = &plan.cards[0];
    let r = Record::new(&PLAN)
        .w("place", place)
        .w("card", &machine.cards[0].name)
        .u("ctx_max", plan.ctx_max)
        .u("card_experts", card.experts)
        .u("card_expert_bytes", card.expert_bytes)
        .u("host_experts", plan.host.experts)
        .u("host_expert_bytes", plan.host.expert_bytes)
        .u("host_shadow", plan.host.shadow_bytes)
        .w(
            "n_l",
            format!(
                "{}..{}",
                held.iter().min().copied().unwrap_or(0),
                held.iter().max().copied().unwrap_or(0)
            ),
        )
        .u("n_l_layers", held.len())
        .w(
            "card_budget",
            plan.card_budget
                .map_or_else(|| "none".to_string(), |b| b.to_string()),
        );
    let r = match machine.cards.first().and_then(|c| c.free_bytes) {
        Some(free) => r.u("card_free", free),
        None => r,
    };
    r.csv("devices", plan_devices(machine))
        .w("cuda_order", cuda_order())
}

/// Each card of `machine` as the `plan` record's `devices` field names it:
/// `stage` or `tier<i>`, its name, its ordinal (`by-name` for a card a
/// census-free plan opens by its name), its usable bytes. The UUID stays in
/// the process: the open checks it, no record prints it.
#[must_use]
pub fn plan_devices(machine: &Machine) -> Vec<String> {
    let role = |i: usize| {
        if i < machine.cards.len() {
            "stage".to_string()
        } else {
            format!("tier{}", i - machine.cards.len())
        }
    };
    machine
        .all_cards()
        .enumerate()
        .map(|(i, c)| {
            let at = c
                .device
                .map_or_else(|| "by-name".to_string(), |d| format!("cuda{}", d.ordinal));
            format!("{}:{}:{at}:{}", role(i), c.name, c.usable_bytes)
        })
        .collect()
}

/// The order this process's CUDA ordinals follow: `CUDA_DEVICE_ORDER`, or
/// the driver's default when it is unset.
#[must_use]
pub fn cuda_order() -> String {
    match std::env::var("CUDA_DEVICE_ORDER") {
        Ok(v) if !v.trim().is_empty() => v.split_whitespace().collect::<Vec<_>>().join("_"),
        _ => "unset(FASTEST_FIRST)".to_string(),
    }
}

/// The load's phases record: `open` the file's headers' wall, `plan` the
/// placement plan's when its loader timed it, then the walls a placed load
/// timed, each in seconds — and `other_s` the rest of `total`, the load
/// line's `load_s`. The walls must not outrun the total: the record refuses,
/// naming the kind, when they do.
#[allow(
    clippy::too_many_arguments,
    reason = "the load phases record's fields, one an argument (rust-quality R8)"
)]
pub fn load_phases(
    total: f64,
    open: f64,
    plan: Option<f64>,
    context: f64,
    upload: f64,
    derive: f64,
    host_set: f64,
    body: f64,
    head: f64,
) -> Record {
    let sum = open + plan.unwrap_or(0.0) + context + upload + derive + host_set + body + head;
    let other = total - sum;
    if other < 0.0 {
        panic!("record load_phases: its phases sum to {sum:.2} s, past the load's {total:.2} s");
    }
    let mut r = Record::new(&LOAD_PHASES).f("open_s", open);
    if let Some(plan) = plan {
        r = r.f("plan_s", plan);
    }
    r.f("context_s", context)
        .f("upload_s", upload)
        .f("derive_s", derive)
        .f("host_set_s", host_set)
        .f("body_s", body)
        .f("head_s", head)
        .f("other_s", other)
}

/// The `nvtier` record of a load's arena: its budget, the paged bytes it
/// serves and the counters since the load. A load that attached no tier has
/// no record — honest absence, never a zeroed row.
#[cfg(feature = "gpu")]
#[must_use]
pub fn nvtier_of(tier: Option<&bloomery_gpu::host::nvtier::NvTier>) -> Option<Record> {
    let tier = tier?;
    let s = tier.stats();
    Some(
        Record::new(&NVTIER)
            .u("budget", tier.budget())
            .u("paged", tier.paged_bytes())
            .u("misses", s.misses)
            .u("fills", s.fills)
            .u("fill_bytes", s.fill_bytes)
            .u("fill_ns", s.fill_ns)
            .u("evictions", s.evictions)
            .u("resident_bytes", s.resident_bytes)
            .u("buffered_reads", s.buffered_reads)
            .u("buffered_bytes", s.buffered_bytes)
            .u("drops", s.drops)
            .u("drop_bytes", s.drop_bytes)
            .u("drop_ns", s.drop_ns),
    )
}

/// A placed load's host-set records: the set read in, with its wall, or not;
/// then the set locked, when it was — each with its bytes in the r8 sidecar.
#[cfg(feature = "gpu")]
pub fn host_residency(h: &HostResidency) -> Vec<Record> {
    let mut out = vec![match h.populated() {
        Some(w) => Record::new(&HOST_POPULATE)
            .u("host_populate_bytes", w.bytes())
            .f("populate_s", w.wall().as_secs_f64())
            .u("sidecar_bytes", w.sidecar_bytes()),
        None => Record::new(&HOST_POPULATE_OFF),
    }];
    if let Some(l) = h.lock() {
        out.push(
            Record::new(&HOST_LOCK)
                .u("host_lock_bytes", l.bytes())
                .u("sidecar_bytes", l.sidecar_bytes()),
        );
    }
    out
}

/// The resolved residency lever's record.
pub fn residency_lever(pick: bloomery_levers::ResidencyPick) -> Record {
    Record::new(&RESIDENCY_LEVER)
        .w("residency", pick.word)
        .w("why", pick.why.name())
}

/// The `draft yield` record of `y` ([`bloomery_levers::DraftYield`]'s own
/// numbers, one owner with the rule that decided them).
pub fn draft_yield(y: &bloomery_levers::DraftYield) -> Record {
    Record::new(&DRAFT_YIELD)
        .u("card_bytes", y.card_bytes)
        .u("with", y.with as u64)
        .u("without", y.without as u64)
        .u("base", y.base as u64)
}

/// A round of several slots' record ([`SLOTS_ROUND`]): `cmd` the command the
/// round ran (`step`, a step a row, or `pass`, a drafted pass a row), `rows`
/// the round's rows, `passes` the passes the seat ran them as, and `slots`
/// the resident sequences it serves.
pub fn slots_round(cmd: &str, rows: usize, passes: usize, slots: usize) -> Record {
    Record::new(&SLOTS_ROUND)
        .w("cmd", cmd)
        .u("rows", rows)
        .u("passes", passes)
        .u("slots", slots)
}

/// The record of what the residency lever unset resolved to in
/// `generate_qwen3moe`.
pub fn residency_unset(pick: &bloomery_levers::Residency38Pick) -> Record {
    Record::new(&RESIDENCY_UNSET)
        .w("residency", pick.word())
        .w("why", pick.why)
}

/// [`residency_unset`] of a plan that pages `nvme_bytes` of routed experts
/// through the NVMe tier's RAM arena of `arena_bytes`: `off`, since a
/// promotion would copy those experts through the file mapping, read cold
/// from the drive after the tier's drops.
pub fn residency_unset_paged(nvme_bytes: u64, arena_bytes: u64) -> Record {
    Record::new(&RESIDENCY_UNSET).w("residency", "off").w(
        "why",
        format_args!(
            "unset: the plan pages {nvme_bytes} B of routed experts through the NVMe tier's RAM \
             arena ({arena_bytes} B), whose promotions would read them cold through the file \
             mapping"
        ),
    )
}

/// The churn pool's record under the residency word `residency`, against
/// the plan it was taken from.
pub fn residency_host(
    residency: &str,
    pool: &model::placement::churn::ChurnPool,
    plan: &Plan<'_>,
) -> Record {
    residency_host_beside(residency, pool, plan, 0)
}

/// [`residency_host`] of a load whose host set also holds `beside` bytes the
/// plan does not count (a draft layer's experts hosted beside the plan's
/// own): `headroom_after` is the plan's headroom less the pool and them, the
/// sum the machine's `ChurnPool::check_beside` holds to 0 or more.
pub fn residency_host_beside(
    residency: &str,
    pool: &model::placement::churn::ChurnPool,
    plan: &Plan<'_>,
    beside: u64,
) -> Record {
    Record::new(&RESIDENCY_HOST)
        .w("residency", residency)
        .u("pinned", pool.pinned)
        .u("churn_experts", pool.experts)
        .u("churn_bytes", pool.bytes)
        .u("headroom", plan.host.headroom_bytes)
        .u(
            "headroom_after",
            pool.headroom_after(plan) - i128::from(beside),
        )
}

/// A residency boundary's record, of a machine driven directly
/// ([`PassKind::Driver`]).
#[cfg(feature = "gpu")]
pub fn residency_pass(r: &PassReport) -> Record {
    residency_pass_of(PassKind::Driver, r)
}

/// A residency boundary's record, ending a pass of `kind`.
#[cfg(feature = "gpu")]
pub fn residency_pass_of(kind: PassKind, r: &PassReport) -> Record {
    let rec = Record::new(&RESIDENCY_PASS)
        .w("pass", kind.word())
        .u("boundary", r.boundary)
        .u("kept", r.kept)
        .u("rows", r.rows)
        .u("landed", r.landed)
        .u("late", r.late)
        .u("made", r.made)
        .u("in_flight", r.in_flight)
        .u("bytes", r.bytes)
        .u("end_us", r.end_us)
        .u("boundary_us", r.boundary_us)
        .u("wait_us", r.wait_us)
        .u("issue_us", r.issue_us)
        .u("stage_us", r.stage_us)
        .u("prepare_us", r.prepare_us)
        .u("rereads", r.rereads)
        .u("reread_bytes", r.reread_bytes)
        .u("reread_us", r.reread_us)
        .u("unresident", r.unresident.count())
        .csv("unresident_last", r.unresident.last().map(|u| u.mark()))
        .u("faulting", r.faulting.count())
        .csv("faulting_first", r.faulting.first().map(|f| f.mark()));
    match r.picked {
        Some(picked) => rec.u("picked", picked),
        None => rec,
    }
}

/// A helper thread's record: `name`, where it asked to run, the cpu it is
/// pinned to, the cpus in its mask.
pub fn helper(
    name: &str,
    asked: impl std::fmt::Display,
    pinned: Option<usize>,
    cpus: usize,
) -> Record {
    Record::new(&HELPER)
        .w("name", name)
        .w("asked", asked)
        .w(
            "pinned",
            pinned.map_or_else(|| "float".to_owned(), |c| c.to_string()),
        )
        .u("cpus", cpus)
}

/// A dropped residency machine's leak record.
#[cfg(feature = "gpu")]
pub fn residency_leak(l: &Leak) -> Record {
    let r = Record::new(&RESIDENCY_LEAK).w("reason", l.reason.word());
    let r = match l.code {
        Some(code) => r.u("code", code),
        None => r,
    };
    r.u("ring", l.ring_bytes).u("words", l.words_bytes)
}

/// A prompt's batch-walk records ([`PromptStats`]): one `stat prompt lb` a
/// layer-batch, then the `stat prompt split` over them — the split's sums are
/// the rows' sums, its enqueue the walks' wall less the serves'.
#[cfg(feature = "gpu")]
pub fn prompt_stats(s: &PromptStats) -> Vec<Record> {
    let ms = |ns: u64| ns as f64 / 1e6;
    let lbs = s.rows.len() as u64;
    let (mut union, mut wait, mut serve, mut enqueue, mut slots) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut experts, mut m_hot, mut cols_hot, mut m_sq) = (0u64, 0u64, 0u64, 0u64);
    let mut m_max = 0usize;
    let (mut front, mut down, mut shadow, mut up, mut back) = (0.0_f64, 0.0, 0.0, 0.0, 0.0);
    let mut out = Vec::with_capacity(s.rows.len() + 1);
    for r in &s.rows {
        union += r.union_ns;
        wait += r.wait_ns;
        serve += r.serve_ns;
        enqueue += r.enqueue_ns;
        slots += r.slots;
        experts += r.experts as u64;
        m_max = m_max.max(r.m_max);
        m_hot += r.m_hot as u64;
        cols_hot += r.cols_hot as u64;
        m_sq += r.m_sq;
        front += r.front_ms;
        down += r.down_ms;
        shadow += r.shadow_ms;
        up += r.upload_ms;
        back += r.back_ms;
        out.push(
            Record::new(&STAT_PROMPT_LB)
                .u("b", r.b)
                .u("layer", r.layer as u64)
                .u("cols", r.cols as u64)
                .u("slots", r.slots)
                .u("experts", r.experts as u64)
                .u("m_max", r.m_max as u64)
                .u("m_hot", r.m_hot as u64)
                .u("cols_hot", r.cols_hot as u64)
                .u("m_sq", r.m_sq)
                .f("wait_ms", ms(r.wait_ns))
                .f("union_ms", ms(r.union_ns))
                .f("serve_ms", ms(r.serve_ns))
                .f("enqueue_ms", ms(r.enqueue_ns))
                .f("card_front_ms", r.front_ms)
                .f("card_down_ms", r.down_ms)
                .f("card_shadow_ms", r.shadow_ms)
                .f("card_upload_ms", r.upload_ms)
                .f("card_back_ms", r.back_ms),
        );
    }
    let per = |v: f64| if lbs == 0 { 0.0 } else { v / lbs as f64 };
    out.push(
        Record::new(&STAT_PROMPT_SPLIT)
            .u("ubatches", s.ubatches)
            .u("layer_batches", lbs)
            .f("prologue_ms", ms(s.prologue_ns))
            .f("walk_ms", ms(s.walk_ns))
            .f("fault_ms", ms(s.fault_ns))
            .f("ckpt_ms", ms(s.ckpt_ns))
            .f("union_ms", ms(union))
            .f("wait_ms", ms(wait))
            .f("serve_ms", ms(serve))
            .f("enqueue_ms", ms(s.walk_ns.saturating_sub(serve)))
            .u("host_slots", slots)
            .f("union_lb", per(ms(union)))
            .f("wait_lb", per(ms(wait)))
            .f("serve_lb", per(ms(serve)))
            .f("enqueue_lb", per(ms(enqueue)))
            .f("slots_lb", per(slots as f64))
            .f("experts_lb", per(experts as f64))
            .u("m_max", m_max as u64)
            .f("m_hot_lb", per(m_hot as f64))
            .f("cols_hot_lb", per(cols_hot as f64))
            .f("m_sq_lb", per(m_sq as f64))
            .f("card_front_ms", front)
            .f("card_down_ms", down)
            .f("card_shadow_ms", shadow)
            .f("card_upload_ms", up)
            .f("card_back_ms", back)
            .f("card_front_lb", per(front))
            .f("card_down_lb", per(down))
            .f("card_shadow_lb", per(shadow))
            .f("card_upload_lb", per(up))
            .f("card_back_lb", per(back)),
    );
    out
}

/// A prompt call's pick record, of group `group`. The optional parts print
/// only where the pick measured them: a refused pick carries `refused`
/// alone, a pick that moved nothing leaves `staged_us` out, and a pick that
/// ran carries its admit and victim columns (0 when it moved nothing).
#[cfg(feature = "gpu")]
pub fn call_stream(group: usize, p: &CallPick) -> Record {
    let r = Record::new(&CALL_STREAM)
        .u("group", group)
        .u("layer", p.layer)
        .u("admitted", p.admitted)
        .u("kept", p.kept)
        .u("bytes", p.bytes)
        .w("counts", format!("{:016x}", p.counts))
        .u("pick_us", p.pick_us)
        .u("backlog_us", p.backlog_us);
    let r = if p.staged_us > 0 {
        r.u("staged_us", p.staged_us)
    } else {
        r
    };
    let r = if p.floor > 0 {
        r.u("floor", p.floor)
    } else {
        r
    };
    let r = if p.refused == 0 {
        r.u("backlog", p.backlog)
    } else {
        r
    };
    let r = if p.fallback > 0 {
        r.u("fallback", u64::from(p.fallback))
    } else {
        r
    };
    if p.refused > 0 {
        r.u("refused", u64::from(p.refused))
    } else {
        r.u("admit_cols", p.admit_cols)
            .u("victim_cols", p.victim_cols)
    }
}

/// A prompt call's end record.
#[cfg(feature = "gpu")]
pub fn call_report(r: &CallReport) -> Record {
    Record::new(&CALL_STREAM_END)
        .u("picks", r.picks)
        .u("admitted", r.admitted)
        .u("bytes", r.bytes)
        .u("pick_us", r.pick_us)
        .u("backlog_us", r.backlog_us)
        .u("kept", u64::from(r.kept))
        .u("restored", r.restored)
        .u("end_us", r.end_us)
        .u("unresident", r.unresident.count())
        .csv("unresident_last", r.unresident.last().map(|u| u.mark()))
        .u("return_us", r.return_us)
}

/// A prompt unit's stream record at one layer, of ubatch `ubatch`.
#[cfg(feature = "gpu")]
pub fn xstream_layer(ubatch: usize, x: &XLayer) -> Record {
    Record::new(&XSTREAM)
        .u("ubatch", ubatch)
        .u("layer", x.layer)
        .u("cols", x.cols)
        .u("host", x.host)
        .u("tail", x.tail)
        .u("streamed", x.streamed)
        .u("streamed_columns", x.streamed_columns)
        .u("host_columns", x.host_columns)
        .u("bytes", x.bytes)
        .u("issue_us", x.issue_us)
        .u("backlog_us", x.backlog_us)
}

/// A prompt call's end record on the expert stream.
#[cfg(feature = "gpu")]
pub fn xstream_end(r: &XReport) -> Record {
    Record::new(&XSTREAM_END)
        .u("layers", r.layers)
        .u("streamed", r.streamed)
        .u("bytes", r.bytes)
        .u("issue_us", r.issue_us)
        .u("backlog_us", r.backlog_us)
        .u("half_slots", r.half_slots)
        .u("staging_slots", r.staging_slots)
        .f("lane_gbs", r.lane_b_per_us / 1000.0)
        .u("end_us", r.end_us)
}

/// A residency reset's record.
#[cfg(feature = "gpu")]
pub fn residency_reset(r: &ResetReport) -> Record {
    Record::new(&RESIDENCY_RESET)
        .u("cancelled", r.cancelled)
        .u("copies", r.copies)
        .u("diff", r.diff)
        .u("dropped_bytes", r.dropped_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every binary's kinds: names unique, each value part named once, a
    /// text part followed by a key, a literal or nothing (the parser reads it
    /// up to what comes next), and every head written.
    #[test]
    fn kinds_are_well_formed() {
        for set in [
            GENERATE_DS41,
            BLOOMERY_CHAT,
            BLOOMERY_SERVE_DS41,
            BLOOMERY_SERVE_QWEN38,
            BLOOMERY_SERVE_QWEN3,
            BLOOMERY_SERVE_GLM,
            GENERATE_GLM5NEXT,
            GENERATE_QWEN3MOE,
            GATE_DEEPSEEK41_PREFILL,
            PROBE_NVREAD,
            GATE_NVTIER,
        ] {
            let mut names: Vec<&str> = set.iter().map(|k| k.name).collect();
            names.sort_unstable();
            names.dedup();
            assert_eq!(names.len(), set.len(), "a kind name twice");
            for k in set {
                assert!(!k.head.is_empty(), "{}: no head", k.name);
                let mut seen: Vec<&str> = k.parts.iter().filter_map(Part::value_name).collect();
                let n = seen.len();
                seen.sort_unstable();
                seen.dedup();
                assert_eq!(seen.len(), n, "{}: a value named twice", k.name);
                for (i, p) in k.parts.iter().enumerate() {
                    if let Part::Key(f) | Part::Pos(f) = p
                        && f.ty == Ty::Text
                    {
                        assert!(
                            matches!(k.parts.get(i + 1), None | Some(Part::Key(_) | Part::Lit(_))),
                            "{}: {} is text followed by a part the parser cannot stop at",
                            k.name,
                            f.name
                        );
                    }
                }
            }
        }
    }

    /// Lines as their readers know them.
    #[test]
    fn renders_the_known_text() {
        let step = Record::new(&TIME_STEP)
            .u("i", 3)
            .flag("warm", true)
            .f("ms", 37.369_84)
            .line();
        assert_eq!(step, "time step 3 warm ms=37.3698");
        let prompt = Record::new(&TIME_PROMPT)
            .u("n", 512)
            .f("ms", 4_285.340_6)
            .f("tok/s", 119.48)
            .u("passes", 1)
            .w("kind", "batch")
            .line();
        assert_eq!(
            prompt,
            "time prompt n=512 ms=4285.3406 tok/s=119.48 passes=1 kind=batch"
        );
        let lock = Record::new(&HOST_LOCK)
            .u("host_lock_bytes", 8192)
            .u("sidecar_bytes", 4096)
            .line();
        assert_eq!(lock, "host_lock=8192 B sidecar_bytes=4096");
        let plan38 = Record::new(&PLAN38)
            .w("place", "a")
            .w("card", "A6000")
            .w("experts", "card")
            .u("ctx_max", 4096)
            .u("host_experts", 11_735)
            .u("card_experts", 12_841)
            .line();
        assert_eq!(
            plan38,
            "plan place=a card=A6000 experts=card ctx_max=4096 host_experts=11735 card_experts=12841"
        );
        let decide = Record::new(&LISTENING_DECIDE)
            .w("model", "m-Q3_K_M.gguf")
            .w("quant", "q4_K×1988 q6_K×97")
            .w("head", "joint_head.safetensors")
            .w("row", "clef")
            .w("addr", "127.0.0.1:39291")
            .line();
        assert_eq!(
            decide,
            "bloomery-serve-decide: model=m-Q3_K_M.gguf quant=q4_K×1988 q6_K×97 \
             head=joint_head.safetensors row=clef listening on http://127.0.0.1:39291"
        );
        let smoke = Record::new(&SMOKE)
            .w("mode", "graph")
            .w("place", "a")
            .u("prompt_tokens", 0)
            .u("depth", 512)
            .u("generated", 2)
            .u("warm", 0)
            .u("steps", 1)
            .f("p50_ms", 37.3698)
            .f("mean_ms", 37.3698)
            .f("tok/s(p50)", 26.76)
            .line();
        assert_eq!(
            smoke,
            "SMOKE mode=graph place=a prompt_tokens=0 depth=512 generated=2 warm=0 steps=1 \
             p50_ms=37.3698 mean_ms=37.3698 tok/s(p50)=26.76"
        );
        let lb = Record::new(&STAT_PROMPT_LB)
            .u("b", 0)
            .u("layer", 3)
            .u("cols", 4096)
            .u("slots", 40960)
            .u("experts", 512)
            .u("m_max", 97)
            .u("m_hot", 0)
            .u("cols_hot", 0)
            .u("m_sq", 3_312_000)
            .f("wait_ms", 21.53)
            .f("union_ms", 214.87)
            .f("serve_ms", 236.71)
            .f("enqueue_ms", 0.94)
            .f("card_front_ms", 19.62)
            .f("card_down_ms", 1.84)
            .f("card_shadow_ms", 0.71)
            .f("card_upload_ms", 1.79)
            .f("card_back_ms", 0.21)
            .line();
        assert_eq!(
            lb,
            "stat prompt lb b=0 layer=3 cols=4096 slots=40960 experts=512 m_max=97 m_hot=0 \
             cols_hot=0 m_sq=3312000 wait_ms=21.53 union_ms=214.87 \
             serve_ms=236.71 enqueue_ms=0.94 card_front_ms=19.62 card_down_ms=1.84 \
             card_shadow_ms=0.71 card_upload_ms=1.79 card_back_ms=0.21"
        );
        let split = Record::new(&STAT_PROMPT_SPLIT)
            .u("ubatches", 1)
            .u("layer_batches", 48)
            .f("prologue_ms", 18.4)
            .f("walk_ms", 13780.9)
            .f("fault_ms", 0.0)
            .f("ckpt_ms", 0.0)
            .f("union_ms", 10313.8)
            .f("wait_ms", 1033.4)
            .f("serve_ms", 11364.1)
            .f("enqueue_ms", 2416.8)
            .u("host_slots", 1966080)
            .f("union_lb", 214.87)
            .f("wait_lb", 21.53)
            .f("serve_lb", 236.75)
            .f("enqueue_lb", 0.94)
            .f("slots_lb", 40960.0)
            .f("experts_lb", 512.0)
            .u("m_max", 97)
            .f("m_hot_lb", 0.0)
            .f("cols_hot_lb", 0.0)
            .f("m_sq_lb", 3312000.0)
            .f("card_front_ms", 941.8)
            .f("card_down_ms", 88.3)
            .f("card_shadow_ms", 34.1)
            .f("card_upload_ms", 85.9)
            .f("card_back_ms", 10.1)
            .f("card_front_lb", 19.62)
            .f("card_down_lb", 1.84)
            .f("card_shadow_lb", 0.71)
            .f("card_upload_lb", 1.79)
            .f("card_back_lb", 0.21)
            .line();
        assert_eq!(
            split,
            "stat prompt split ubatches=1 layer_batches=48 prologue_ms=18.4 walk_ms=13780.9 \
             fault_ms=0.0 ckpt_ms=0.0 union_ms=10313.8 wait_ms=1033.4 serve_ms=11364.1 enqueue_ms=2416.8 \
             host_slots=1966080 union_lb=214.87 wait_lb=21.53 serve_lb=236.75 enqueue_lb=0.94 \
             slots_lb=40960.0 experts_lb=512.0 m_max=97 m_hot_lb=0.0 cols_hot_lb=0.0 \
             m_sq_lb=3312000.0 card_front_ms=941.8 card_down_ms=88.3 card_shadow_ms=34.1 \
             card_upload_ms=85.9 card_back_ms=10.1 card_front_lb=19.62 card_down_lb=1.84 \
             card_shadow_lb=0.71 card_upload_lb=1.79 card_back_lb=0.21"
        );
    }

    /// The checked-in schemas `records.py` reads are the binaries' own: a
    /// kind changed here fails until `just records-refresh` rewrites them.
    #[test]
    fn checked_in_schemas_are_current() {
        let files = [
            (
                "generate_ds41",
                GENERATE_DS41,
                include_str!("../../../tools/bloomery/schema/generate_ds41.jsonl"),
            ),
            (
                "bloomery-chat",
                BLOOMERY_CHAT,
                include_str!("../../../tools/bloomery/schema/bloomery-chat.jsonl"),
            ),
            (
                "bloomery-serve-ds41",
                BLOOMERY_SERVE_DS41,
                include_str!("../../../tools/bloomery/schema/bloomery-serve-ds41.jsonl"),
            ),
            (
                "bloomery-serve-qwen38",
                BLOOMERY_SERVE_QWEN38,
                include_str!("../../../tools/bloomery/schema/bloomery-serve-qwen38.jsonl"),
            ),
            (
                "generate_glm5next",
                GENERATE_GLM5NEXT,
                include_str!("../../../tools/bloomery/schema/generate_glm5next.jsonl"),
            ),
            (
                "clef_hidden",
                CLEF_HIDDEN_BIN,
                include_str!("../../../tools/bloomery/schema/clef_hidden.jsonl"),
            ),
            (
                "generate_qwen3moe",
                GENERATE_QWEN3MOE,
                include_str!("../../../tools/bloomery/schema/generate_qwen3moe.jsonl"),
            ),
            (
                "gate_deepseek41_prefill",
                GATE_DEEPSEEK41_PREFILL,
                include_str!("../../../tools/bloomery/schema/gate_deepseek41_prefill.jsonl"),
            ),
            (
                "probe_nvread",
                PROBE_NVREAD,
                include_str!("../../../tools/bloomery/schema/probe_nvread.jsonl"),
            ),
        ];
        for (bin, kinds, file) in files {
            assert!(
                schema(bin, kinds) == file,
                "tools/bloomery/schema/{bin}.jsonl is not {bin}'s schema: run just records-refresh"
            );
        }
    }

    #[test]
    #[should_panic(expected = "record time_step: ms given where i is due")]
    fn refuses_a_value_out_of_order() {
        let _ = Record::new(&TIME_STEP).f("ms", 1.0).line();
    }

    #[test]
    #[should_panic(expected = "record fed: last left out")]
    fn refuses_a_line_with_a_part_missing() {
        let _ = Record::new(&FED)
            .u("ids", 4)
            .list("first", &[1u32, 2])
            .line();
    }

    /// Under a binary's registered kinds, one of them renders and a line of
    /// another kind is refused, naming the kind and the binary.
    #[test]
    #[should_panic(expected = "record listening: generate_ds41 does not print this kind")]
    fn refuses_a_kind_its_binary_does_not_print() {
        let generate = Prints {
            bin: "generate_ds41",
            kinds: GENERATE_DS41,
        };
        let lock = Record::new(&HOST_LOCK)
            .u("host_lock_bytes", 4096)
            .u("sidecar_bytes", 0)
            .line_under(Some(&generate));
        assert_eq!(lock, "host_lock=4096 B sidecar_bytes=0");
        let _ = Record::new(&LISTENING)
            .w("place", "a")
            .u("ctx", 4096)
            .w("addr", "127.0.0.1:8080")
            .line_under(Some(&generate));
    }

    // ------------------------------------------------------------ the reader

    /// Lines the servers printed, as the serve gates' logs carry them.
    const PLAN38_PLACED: &str = "plan place=cuda0 card=3090 arch=qwen35moe experts=card \
        ctx_max=1088 host_experts=5427 card_experts=4813 n_l=120-121 card_free=25052381184 \
        devices=[stage:3090:cuda0:25350373376] cuda_order=unset(FASTEST_FIRST)";
    const PLAN38_NO_FREE: &str = "plan place=cuda0 card=3090 arch=qwen35moe experts=card \
        ctx_max=1088 host_experts=5427 card_experts=4813 n_l=120-121 \
        devices=[stage:3090:cuda0:25350373376] cuda_order=unset(FASTEST_FIRST)";
    const PLAN_BP: &str = "plan place=bp card=A6000 ctx_max=16384 card_experts=2552 \
        (38801506304 B) host_experts=8045 (123885584384 B) host_shadow=0 B n_l=65..66 on 39 \
        layers card_budget=none card_free=50668240896 \
        devices=[stage:A6000:cuda1:50952404992,tier0:3090:cuda0:25350373376] \
        cuda_order=unset(FASTEST_FIRST)";
    const RESIDENCY_HOST_GLM: &str = "residency host residency=mid-p0-s1 pinned=0 \
        churn_experts=2552 churn_bytes=38801506304 headroom=111730458624 \
        headroom_after=68550098944";
    const LOAD_QWEN3_LINE: &str = "load arch=qwen3moe resident_bytes=24165631476 cache=f16 \
        ctx=23552 slots=2 slot_ctx=23552 layers=48 ubatch=4096 graph_nodes=604 in 1.5 s";
    /// A `residency pass` line of an older text: the reread fields came after it.
    const PASS_OLD: &str = "residency pass pass=driver boundary=119 kept=1 landed=3 late=0 \
        made=0 in_flight=0 bytes=0 end_us=0 boundary_us=4 wait_us=0 issue_us=0 stage_us=0 \
        prepare_us=0";
    /// Lines under a kind's head that are no record: a placement's and the
    /// host tier's own text.
    const TEXT_UNDER_HEADS: &str = "load host_tier r8=on (/models/m-r8/m-r8.gguf)\n\
        load host_tier type=q3_K k=5120 path=r8\n\
        plan card bytes 23113320896 (dense 3891325376 + experts 19221995520) host bytes \
        239830005760\n\
        plan (b′): A6000 (GPU1) 48642009536 B, tier 3090 (GPU0) 881 experts 14777118720 B";

    /// Every kind of every binary's list, rendered by [`Record`] with every
    /// part given (each flag set) and with every optional part left out, reads
    /// back whole under that list as that kind, each value as rendered — text
    /// and words that are not ASCII included.
    #[test]
    fn reads_every_kind_whole() {
        for set in [
            GENERATE_DS41,
            BLOOMERY_CHAT,
            BLOOMERY_SERVE_DS41,
            BLOOMERY_SERVE_QWEN38,
            BLOOMERY_SERVE_QWEN3,
            BLOOMERY_SERVE_GLM,
            GENERATE_GLM5NEXT,
            GENERATE_QWEN3MOE,
            CLEF_HIDDEN_BIN,
            GATE_DEEPSEEK41_PREFILL,
            PROBE_NVREAD,
            GATE_NVTIER,
        ] {
            for &kind in set {
                for full in [true, false] {
                    let mut r = Record::new(kind);
                    let mut want: Vec<(&str, String)> = Vec::new();
                    for p in kind.parts {
                        match p {
                            Part::Lit(_) => {}
                            Part::Flag(n) => {
                                if full {
                                    r = r.flag(n, true);
                                    want.push((n, String::new()));
                                }
                            }
                            Part::Key(f) | Part::Pos(f) => {
                                if f.opt && !full {
                                    continue;
                                }
                                let (next, text) = match f.ty {
                                    Ty::U64 => (r.u(f.name, 7), "7".to_owned()),
                                    Ty::I64 => (r.u(f.name, -3), "-3".to_owned()),
                                    Ty::F64(d) => (r.f(f.name, 1.5), format!("{:.d$}", 1.5)),
                                    Ty::Bool => (r.w(f.name, "true"), "true".to_owned()),
                                    Ty::Word => (r.w(f.name, "w×1"), "w×1".to_owned()),
                                    Ty::Text => (r.w(f.name, "a b′ c"), "a b′ c".to_owned()),
                                    Ty::List => (r.list(f.name, &[1u32, 2]), "[1, 2]".to_owned()),
                                    Ty::Csv => (r.csv(f.name, ["a", "b′"]), "[a,b′]".to_owned()),
                                };
                                r = next;
                                want.push((f.name, text));
                            }
                        }
                    }
                    let line = r.line();
                    let got = Log::of(&line, set)
                        .one(kind)
                        .unwrap_or_else(|e| panic!("{}: {e}", kind.name));
                    assert_eq!(got.values, want, "{}: {line}", kind.name);
                    for (name, text) in &want {
                        let Some(Part::Key(f) | Part::Pos(f)) =
                            kind.parts.iter().find(|p| p.value_name() == Some(name))
                        else {
                            assert_eq!(got.flag(name), Ok(true), "{}", kind.name);
                            continue;
                        };
                        let read = match f.ty {
                            Ty::U64 => got.u64(name).map(|v| v.to_string()),
                            Ty::I64 => got.i64(name).map(|v| v.to_string()),
                            Ty::F64(d) => got.f64(name).map(|v| format!("{v:.d$}")),
                            Ty::Bool => got.bool(name).map(|v| v.to_string()),
                            Ty::Word => got.word(name).map(str::to_owned),
                            Ty::Text => got.text(name).map(str::to_owned),
                            Ty::List => Ok(text.clone()),
                            Ty::Csv => got.csv(name).map(|v| format!("[{}]", v.join(","))),
                        };
                        assert_eq!(read.as_ref(), Ok(text), "{}.{name}: {line}", kind.name);
                    }
                }
            }
        }
    }

    /// An optional part the line leaves out reads as absent, and the parts
    /// after it keep their own values: the server's `plan` lines with and
    /// without `card_free`, and a qwen4exp `plan` line with no `n_l`.
    #[test]
    fn optional_parts_left_out_read_as_absent() {
        let q3 = |line: &str| Log::of(line, BLOOMERY_SERVE_QWEN3).one(&PLAN38).unwrap();
        let placed = q3(PLAN38_PLACED);
        assert_eq!(placed.opt_u64("card_free"), Ok(Some(25_052_381_184)));
        assert_eq!(placed.word("n_l"), Ok("120-121"));
        let bare = q3(PLAN38_NO_FREE);
        assert_eq!(bare.opt_u64("card_free"), Ok(None));
        assert_eq!(bare.opt_word("tier"), Ok(None));
        assert_eq!(bare.opt_u64("tier_experts"), Ok(None));
        assert_eq!(bare.opt_word("why"), Ok(None));
        assert_eq!(
            bare.csv("devices"),
            Ok(vec!["stage:3090:cuda0:25350373376"])
        );
        assert_eq!(bare.word("cuda_order"), Ok("unset(FASTEST_FIRST)"));
        assert!(matches!(
            bare.u64("card_free"),
            Err(ReadError::Missing {
                kind: "plan38",
                field: "card_free",
                ..
            })
        ));
        let exp = Record::new(&PLAN38)
            .w("place", "a")
            .w("card", "A6000")
            .w("experts", "card")
            .u("ctx_max", 4096)
            .u("host_experts", 11_735)
            .u("card_experts", 12_841)
            .u("card_free", 5)
            .line();
        let exp = Log::of(&exp, BLOOMERY_SERVE_QWEN38).one(&PLAN38).unwrap();
        assert_eq!(exp.opt_word("n_l"), Ok(None));
        assert_eq!(exp.opt_word("arch"), Ok(None));
        assert_eq!(exp.u64("card_free"), Ok(5));
        let bp = Log::of(PLAN_BP, BLOOMERY_SERVE_DS41).one(&PLAN).unwrap();
        assert_eq!(bp.u64("card_free"), Ok(50_668_240_896));
        assert_eq!(
            bp.csv("devices"),
            Ok(vec![
                "stage:A6000:cuda1:50952404992",
                "tier0:3090:cuda0:25350373376"
            ])
        );
        assert_eq!(bp.word("card_budget"), Ok("none"));
    }

    /// Kinds that share a head are told apart by the whole line: a qwen3moe
    /// `plan` line is a `plan38` record beside a `plan` one, and under the
    /// V4.1 server's kinds a `cache save` line is a `cache_save` record, not
    /// the `cache` (config) one, nor is a `cache reuse` line.
    #[test]
    fn shared_heads_read_by_the_whole_line() {
        let both: &[&'static Kind] = &[&PLAN, &PLAN38];
        let log = Log::of(&format!("{PLAN_BP}\n{PLAN38_PLACED}"), both);
        assert_eq!(
            log.one(&PLAN).map(|f| f.line().to_owned()),
            Ok(PLAN_BP.to_owned())
        );
        let plan38 = log.one(&PLAN38).unwrap();
        assert_eq!((plan38.at(), plan38.line()), (2, PLAN38_PLACED));
        let ds41 = Log::of(
            "cache ram=8589934592 headroom=111730458624 user_start=<|User|> in_template=true\n\
             cache save positions=46 bytes=359555072 ms=103.234 entries=2 cache_bytes=718986240\n\
             cache reuse common=40 ask=40 kept=38 held=46 reason=the cut at a message start",
            BLOOMERY_SERVE_DS41,
        );
        let config = ds41.one(&CACHE_CONFIG).unwrap();
        assert_eq!(config.u64("ram"), Ok(8_589_934_592));
        assert_eq!(config.bool("in_template"), Ok(true));
        assert_eq!(
            ds41.one(&CACHE_SAVE).and_then(|f| f.u64("positions")),
            Ok(46)
        );
        assert_eq!(
            ds41.one(&CACHE_REUSE)
                .map(|f| f.text("reason").map(str::to_owned)),
            Ok(Ok("the cut at a message start".to_owned()))
        );
        let q3 = Log::of(LOAD_QWEN3_LINE, BLOOMERY_SERVE_QWEN3);
        assert_eq!(
            q3.one(&LOAD_QWEN3).and_then(|f| f.u64("slot_ctx")),
            Ok(23_552)
        );
    }

    /// A line that opens a kind's record and does not read whole is refused
    /// by name on every read of each kind it opens, never read as no record;
    /// a line under a kind's head that opens no record (a placement's text,
    /// the host tier's) refuses nothing.
    #[test]
    fn a_line_that_opens_a_record_and_does_not_read_whole_is_refused() {
        let text = format!("{RESIDENCY_HOST_GLM}\n{PASS_OLD}");
        let log = Log::of(&text, BLOOMERY_SERVE_QWEN38).named("server.err");
        let broken = ReadError::Broken {
            log: "server.err".to_owned(),
            kind: "residency_pass",
            at: 2,
            line: PASS_OLD.to_owned(),
        };
        assert_eq!(log.all(&RESIDENCY_PASS).unwrap_err(), broken);
        assert_eq!(log.first(&RESIDENCY_PASS).unwrap_err(), broken);
        assert_eq!(log.one(&RESIDENCY_PASS).unwrap_err(), broken);
        assert!(
            broken
                .to_string()
                .contains("server.err: line 2 opens a `residency_pass`")
        );
        assert_eq!(
            log.one(&RESIDENCY_HOST).and_then(|f| f.u64("pinned")),
            Ok(0)
        );
        // `plan` and `plan38` share their head and first field: a line that
        // opens both and reads as neither refuses both.
        let both: &[&'static Kind] = &[&PLAN, &PLAN38];
        let torn = "plan place=a card=A6000 ctx_max=4096 card_experts=12841";
        let log = Log::of(torn, both);
        assert!(matches!(
            log.first(&PLAN),
            Err(ReadError::Broken { kind: "plan", .. })
        ));
        assert!(matches!(
            log.first(&PLAN38),
            Err(ReadError::Broken { kind: "plan38", .. })
        ));
        let text = format!("{TEXT_UNDER_HEADS}\n{PLAN_BP}");
        let log = Log::of(&text, BLOOMERY_SERVE_DS41);
        assert_eq!(
            log.one(&PLAN).map(|f| f.line().to_owned()),
            Ok(PLAN_BP.to_owned())
        );
        assert_eq!(log.first(&LOAD_GENERATOR).map(|f| f.is_none()), Ok(true));
    }

    /// `one` names the kind and the log when the log holds none of it, or
    /// more than one, and `all` refuses a kind the log is not read by.
    #[test]
    fn one_names_an_absent_kind_and_the_log() {
        let log = Log::of(TEXT_UNDER_HEADS, BLOOMERY_SERVE_DS41).named("server.err");
        let absent = log.one(&PLAN).unwrap_err();
        assert_eq!(
            absent,
            ReadError::Absent {
                log: "server.err".to_owned(),
                kind: "plan"
            }
        );
        assert_eq!(absent.to_string(), "server.err: no `plan` record");
        let twice = Log::of(&format!("{PLAN_BP}\n{PLAN_BP}"), BLOOMERY_SERVE_DS41);
        assert!(matches!(
            twice.one(&PLAN),
            Err(ReadError::Many {
                kind: "plan",
                n: 2,
                ..
            })
        ));
        assert!(matches!(
            log.all(&LISTENING38),
            Err(ReadError::NotPrinted {
                kind: "listening38",
                ..
            })
        ));
    }

    /// A field read is refused by name when the kind has no such field, when
    /// the field is of another type, when the line left it out, and when its
    /// value is past the type's range.
    #[test]
    fn field_reads_refuse_by_name() {
        let host = Log::of(RESIDENCY_HOST_GLM, BLOOMERY_SERVE_QWEN38)
            .one(&RESIDENCY_HOST)
            .unwrap();
        assert_eq!(host.i64("headroom_after"), Ok(68_550_098_944));
        assert_eq!(host.word("residency"), Ok("mid-p0-s1"));
        assert!(matches!(
            host.u64("landed"),
            Err(ReadError::NoField {
                kind: "residency_host",
                ..
            })
        ));
        assert!(matches!(
            host.u64("headroom"),
            Err(ReadError::Type {
                field: "headroom",
                ty: Ty::I64,
                asked: "u64",
                ..
            })
        ));
        assert!(matches!(
            host.csv("residency"),
            Err(ReadError::Type {
                ty: Ty::Word,
                asked: "csv",
                ..
            })
        ));
        let plan = Log::of(PLAN38_NO_FREE, BLOOMERY_SERVE_QWEN3)
            .one(&PLAN38)
            .unwrap();
        assert!(matches!(
            plan.u64("card_free"),
            Err(ReadError::Missing {
                field: "card_free",
                ..
            })
        ));
        let past = PLAN38_PLACED.replace("card_experts=4813", "card_experts=99999999999999999999");
        let past = Log::of(&past, BLOOMERY_SERVE_QWEN3).one(&PLAN38).unwrap();
        assert!(matches!(
            past.u64("card_experts"),
            Err(ReadError::Value {
                field: "card_experts",
                ..
            })
        ));
    }
}
