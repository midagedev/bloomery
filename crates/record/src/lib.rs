//! The one owner of a record line's syntax: its [`Kind`].
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
//! is every line the binary can print. A shared module's lines are a group of
//! kinds, composed into the list of every binary that enters the module, and
//! the module's entry names it ([`require`]): a binary that enters a module
//! without having registered its group panics by name at its start, whatever
//! flag would have reached the print. [`Log`] reads a log by the same rules
//! in Rust, a record's values by field name ([`Fields`]), and refuses by name
//! ([`ReadError`]) a record absent where one is due, a line that opens a
//! kind's record and does not read whole, and a field the kind lacks, of
//! another type, or left out.

use std::fmt::{Debug, Display, Write as _};
use std::sync::{Mutex, OnceLock, PoisonError};

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
    pub fn value_name(&self) -> Option<&'static str> {
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
pub struct Prints {
    pub bin: &'static str,
    pub kinds: &'static [&'static Kind],
}

impl Prints {
    fn holds(&self, kind: &Kind) -> bool {
        self.kinds.iter().any(|k| std::ptr::eq(*k, kind))
    }
}

/// The process's [`Prints`], set once by [`at_main`]; unset, a line of any
/// kind renders.
static PRINTS: OnceLock<Prints> = OnceLock::new();

/// A group a shared module's entry needs registered ([`require`]) and the entry's name.
#[derive(Clone, Copy)]
pub struct Needs {
    group: &'static [&'static Kind],
    who: &'static str,
}

/// The groups required before [`at_main`] registered, which it checks when it does.
static PENDING: Mutex<Vec<Needs>> = Mutex::new(Vec::new());

/// With `--records-schema` among the process's arguments, print `bin`'s
/// kinds ([`print_schema`]) and exit 0; otherwise register them, after which
/// [`Record::line`] refuses a line of any other kind, and check the groups
/// [`require`] was given before this call. Called right after
/// `bloomery_levers::at_main`, before any argument parse or load; a second
/// call with another binary or list panics by name.
pub fn at_main(bin: &'static str, kinds: &'static [&'static Kind]) {
    if std::env::args_os().skip(1).any(|a| a == "--records-schema") {
        print_schema(bin, kinds);
        std::process::exit(0);
    }
    register_in(&PRINTS, &PENDING, bin, kinds);
}

/// A shared module's entry `who` needs `group` registered: the kinds of its
/// path of prints. Registered already, a kind of `group` missing
/// from the process's list panics by name; before [`at_main`], the group
/// waits and `at_main` checks it. A process that never calls `at_main` (a
/// gate binary) never reads the wait.
pub fn require(group: &'static [&'static Kind], who: &'static str) {
    require_in(&PRINTS, &PENDING, group, who);
}

/// The names of the kinds of `group` that `list` does not hold, in the group's order.
pub fn missing(list: &[&Kind], group: &[&'static Kind]) -> Vec<&'static str> {
    group
        .iter()
        .filter(|g| !list.iter().any(|k| std::ptr::eq(*k, **g)))
        .map(|g| g.name)
        .collect()
}

/// What `held` says when the entry `who` needs `group`: `None` when it holds
/// every kind of it.
fn refusal(held: &Prints, group: &[&'static Kind], who: &str) -> Option<String> {
    let lacking = missing(held.kinds, group);
    (!lacking.is_empty()).then(|| {
        format!(
            "{} enters {who} without registering its records: {}",
            held.bin,
            lacking.join(", ")
        )
    })
}

/// Panics by name when `held` lacks a kind of `group`, the entry `who` needing it.
fn enforce(held: &Prints, group: &[&'static Kind], who: &str) {
    if let Some(why) = refusal(held, group, who) {
        panic!("{why}");
    }
}

/// [`require`] under the registration `slot` and the wait `pending`. Once the
/// process registered, the check takes no lock: an entry that runs per request
/// pays a few pointer compares.
pub fn require_in(
    slot: &OnceLock<Prints>,
    pending: &Mutex<Vec<Needs>>,
    group: &'static [&'static Kind],
    who: &'static str,
) {
    if let Some(held) = slot.get() {
        enforce(held, group, who);
        return;
    }
    let mut wait = pending.lock().unwrap_or_else(PoisonError::into_inner);
    // `at_main` may have registered between the look above and the lock; its drain
    // comes after the lock, so a group pushed here is seen by it.
    if let Some(held) = slot.get() {
        drop(wait);
        enforce(held, group, who);
        return;
    }
    let same = |n: &Needs| {
        n.who == who
            && n.group.len() == group.len()
            && n.group.iter().zip(group).all(|(a, b)| std::ptr::eq(*a, *b))
    };
    if !wait.iter().any(same) {
        wait.push(Needs { group, who });
    }
}

/// [`at_main`]'s registration of `bin`'s `kinds` in `slot`, then the check of
/// every group that waited in `pending`.
pub fn register_in(
    slot: &OnceLock<Prints>,
    pending: &Mutex<Vec<Needs>>,
    bin: &'static str,
    kinds: &'static [&'static Kind],
) {
    let held = slot.get_or_init(|| Prints { bin, kinds });
    assert!(
        held.bin == bin && std::ptr::eq(held.kinds, kinds),
        "record::at_main({bin}): the process registered {}'s kinds already",
        held.bin
    );
    let due = std::mem::take(&mut *pending.lock().unwrap_or_else(PoisonError::into_inner));
    for n in due {
        enforce(held, n.group, n.who);
    }
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
    pub fn line_under(mut self, prints: Option<&Prints>) -> String {
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
        self.emit(false);
    }

    /// The line on stderr, and to the [`tee`] when one is set.
    pub fn eprint(self) {
        self.emit(true);
    }

    /// The line on stderr (`err`, and to the [`tee`] when one is set) or on
    /// stdout, handed back for a caller that keeps what it printed.
    pub fn emit(self, err: bool) -> String {
        let line = self.line();
        if err {
            eprintln!("{line}");
            if let Some(tee) = TEE.get() {
                tee(&line);
            }
        } else {
            println!("{line}");
        }
        line
    }
}

/// Where [`Record::eprint`] also hands each line, set once by [`tee`].
static TEE: OnceLock<fn(&str)> = OnceLock::new();

/// Hands every line [`Record::eprint`] prints from now on to `to` as well: a
/// serving seat's start screen (`serve::screen`) when the seat armed it. A
/// second call keeps the first.
pub fn tee(to: fn(&str)) {
    let _ = TEE.set(to);
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
    /// The parsed field names and values, in record order.
    #[must_use]
    pub fn values(&self) -> &[(&'static str, String)] {
        &self.values
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    /// A kind the core builds itself: a count, a literal, a float and a flag.
    static CORE_KIND: Kind = Kind {
        name: "core_kind",
        head: "core kind",
        parts: &[
            Part::Key(Field {
                name: "count",
                ty: Ty::U64,
                unit: "",
                opt: false,
            }),
            Part::Lit(" in "),
            Part::Pos(Field {
                name: "wall",
                ty: Ty::F64(2),
                unit: "s",
                opt: false,
            }),
            Part::Flag("warm"),
        ],
        doc: "the core's own kind.",
    };

    /// The lines the [`tee`] took.
    static TAKEN: Mutex<Vec<String>> = Mutex::new(Vec::new());

    fn take(line: &str) {
        TAKEN.lock().unwrap().push(line.to_string());
    }

    #[test]
    fn a_value_of_another_type_is_refused_by_name() {
        let err = std::panic::catch_unwind(|| Record::new(&CORE_KIND).u("count", "x").line())
            .expect_err("a word into a u64 part must be refused");
        let msg = err
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| err.downcast_ref::<&str>().copied())
            .unwrap_or_default();
        assert!(
            msg.contains("count = \"x\" is not a u64"),
            "the refusal names the part and the type: {msg}"
        );
    }

    #[test]
    fn a_line_printed_to_stderr_reaches_the_tee() {
        tee(take);
        Record::new(&CORE_KIND)
            .u("count", 3u64)
            .f("wall", 2.0)
            .flag("warm", true)
            .eprint();
        let taken = TAKEN.lock().unwrap();
        assert_eq!(taken.len(), 1, "one line to the tee");
        assert_eq!(taken[0], "core kind count=3 in 2.00 warm");
    }
}
