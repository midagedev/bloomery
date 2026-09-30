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
//! is every line the binary can print.
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
use bloomery_gpu::hybrid::HostResidency;
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

// ---------------------------------------------------------------- the kinds

use Ty::{Bool, Csv, F64, I64, List, Text, U64, Word};

/// Where the placement puts the experts, before the load.
pub static PLAN: Kind = Kind {
    name: "plan",
    head: "plan",
    doc: "The placement the engine is about to load by: its card, context, and where the experts sit.",
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
        key("hot_list", Text, ""),
    ],
};

/// Where a qwen4exp placement puts the routed experts, before the load.
pub static PLAN38: Kind = Kind {
    name: "plan38",
    head: "plan",
    doc: "The qwen4exp placement the engine is about to load by: its card, the expert rule (host or card), the context, where the routed experts sit, and the hot list file the card's experts were ranked by (none: each layer's id prefix).",
    parts: &[
        key("place", Word, ""),
        key("card", Word, ""),
        key("experts", Word, ""),
        key("ctx_max", U64, "positions"),
        key("host_experts", U64, "experts"),
        key("card_experts", U64, "experts"),
        key("hot_list", Text, ""),
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
    doc: "The load a decode loop opened: device bytes, the context, the body's fields, the cards it loaded (a V4.1 binary's; an expert tier card's experts and resident bytes), the step mode and the pin.",
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
    doc: "An arm of one load's --arm list, before its lines: its index, the list's length, its feed (lcg or a corpus), its fed ids and its generated count.",
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

/// One layer-batch of a Qwen3.8 ubatch walk.
pub static STAT_PROMPT38_LB: Kind = Kind {
    name: "stat_prompt38_lb",
    head: "stat prompt38 lb",
    doc: "One layer-batch of a Qwen3.8 ubatch walk under BLOOMERY_STEP_STATS: its columns, the \
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

/// Where a Qwen3.8 prompt's ubatch walks spent their time.
pub static STAT_PROMPT38_SPLIT: Kind = Kind {
    name: "stat_prompt38_split",
    head: "stat prompt38 split",
    doc: "A Qwen3.8 prompt's ubatch walks under BLOOMERY_STEP_STATS, summed: the ubatches and \
          layer-batches that ran, the host prologue every walk's plan took (the PLE rows, the \
          record, their copy) and the walks' whole wall, the serves' union and wait and the host \
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
    doc: "A residency boundary: what the pass before it was (none, step, pair, prompt — a \
          prompt call's rows are not counted — abandoned, or driver: the machine's own test \
          driver), its passes since the load or the last reset, the rows the pass before it kept, the flips that went live there (late: their copies had not completed and \
          the engine stream waited), the flips the rule made, the flips in flight after it, and \
          the bytes its flips copy; the host's microseconds folding the pass into the rule and in \
          the whole boundary call, of \
          them waiting for the landing jobs' staging and issuing the new flips' copies, and the staging thread's since the last \
          boundary copying into the ring and preparing victims.",
    parts: &[
        key("pass", Word, ""),
        key("boundary", U64, ""),
        key("kept", U64, ""),
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

/// Adaptive residency's host share at the plan
/// ([`model::placement::churn::ChurnPool`]).
pub static RESIDENCY_HOST: Kind = Kind {
    name: "residency_host",
    head: "residency host",
    doc: "Adaptive residency's host share, from the plan before the load: the lever's word, the \
          seed experts a layer kept on the stage card, the churn pool (the stage card's experts \
          past them, which the load's host set holds too) and its bytes, and the plan's host \
          headroom before and after the pool.",
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
          to take in the call's earlier jobs.",
    parts: &[
        key("group", U64, ""),
        key("layer", U64, ""),
        key("admitted", U64, ""),
        key("kept", U64, ""),
        key("bytes", U64, "B"),
        key("counts", Word, ""),
        key("pick_us", U64, "us"),
        key("backlog_us", U64, "us"),
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
          after it (1) or went back to the call's start (0) with the experts copied back, and \
          the host's microseconds in the end.",
    parts: &[
        key("picks", U64, ""),
        key("admitted", U64, ""),
        key("bytes", U64, "B"),
        key("pick_us", U64, "us"),
        key("backlog_us", U64, "us"),
        key("kept", U64, ""),
        key("restored", U64, ""),
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

/// One generated step's host-tier counters on a body with no engram rows.
pub static STAT_STEP_HOST: Kind = Kind {
    name: "stat_step_host",
    head: "stat step",
    doc: "BLOOMERY_STEP_STATS: one generated step's host-tier counters, page faults and free device bytes (generate_qwen3moe's qwen4exp body).",
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
    ],
};

/// The steps' summary on a body with no engram rows.
pub static STAT_SUMMARY_HOST: Kind = Kind {
    name: "stat_summary_host",
    head: "stat summary",
    doc: "BLOOMERY_STEP_STATS: the kept steps' host-tier, fault and device-memory statistics (generate_qwen3moe's qwen4exp body).",
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
    doc: "BLOOMERY_DRAFT: proposals, accepts, positions and passes over the run, and the kept passes' positions per second.",
    parts: &[
        key("proposals", U64, ""),
        key("accepts", U64, ""),
        key("positions", U64, "positions"),
        key("passes", U64, ""),
        key("tok/s(positions)", F64(2), "tok/s"),
        key("kind", Word, ""),
    ],
};

/// The MTP draft's window summary.
pub static MTP_SUMMARY: Kind = Kind {
    name: "mtp_summary",
    head: "mtp summary",
    doc: "BLOOMERY_DRAFT=mtp: the windows' proposals and the rows each kept (a count a kept \
          length, 1 to 4), the positions and passes over the run, and the kept passes' \
          positions per second.",
    parts: &[
        key("proposals", U64, ""),
        key("kept", List, "windows"),
        key("positions", U64, "positions"),
        key("passes", U64, ""),
        key("tok/s(positions)", F64(2), "tok/s"),
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
    doc: "The Qwen3.8 server's placement, context and the address it listens on.",
    parts: &[
        key("place", Word, ""),
        key("ctx", U64, "positions"),
        lit(" listening on http://"),
        pos("addr", Word, ""),
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

/// What `bloomery-serve-ds41` prints, all on stderr.
pub static BLOOMERY_SERVE_DS41: &[&Kind] = &[
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
];

/// What `bloomery-serve-qwen38` prints, all on stderr.
pub static BLOOMERY_SERVE_QWEN38: &[&Kind] = &[&PLAN38, &LISTENING38, &CACHE_REUSE];

/// What `generate_glm5next` prints, in the order it prints them.
pub static GENERATE_GLM5NEXT: &[&Kind] = &[
    &PLAN,
    &LOAD_GENERATOR,
    &LOAD_PHASES,
    &HOST_POPULATE,
    &HOST_POPULATE_OFF,
    &HOST_LOCK,
    &CAPTURE,
    &FED,
    &STEP0,
    &TIME_PROMPT,
    &STEP,
    &TIME_STEP,
    &TOKENS,
    &LOGITS,
    &SMOKE,
    &RESIDENCY_PASS,
    &RESIDENCY_RESET,
    &RESIDENCY_LEAK,
];

/// What `generate_qwen3moe` prints as records: under `--dump-taps`, after
/// its `load` line, and under `BLOOMERY_STEP_STATS`, after a qwen4exp run's
/// lines; the binary's other lines are its own.
pub static GENERATE_QWEN3MOE: &[&Kind] = &[
    &PLAN38,
    &TAPS_SEQ,
    &TAPS_DUMP,
    &MTP_SUMMARY,
    &STAT_STEP_HOST,
    &STAT_SUMMARY_HOST,
    &STAT_PROMPT38_SPLIT,
    &STAT_PROMPT38_LB,
];

/// The record `gate_deepseek41_prefill` prints inside its own lines, after
/// its name and the case's.
pub static GATE_DEEPSEEK41_PREFILL: &[&Kind] = &[&STAT_PREFILL_SPLIT];

/// The `plan` record of `plan`, made over `machine` by the placement named
/// `place`: its first card, the experts on it and on the host, the per-layer
/// card counts' range, the budget, and the hot list file the card's experts
/// were ranked by (`none` without one).
pub fn plan(place: &str, machine: &Machine, plan: &Plan<'_>, hot_list: &str) -> Record {
    let held: Vec<u64> = plan.n_l.iter().copied().filter(|&n| n > 0).collect();
    let card = &plan.cards[0];
    Record::new(&PLAN)
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
        )
        .w("hot_list", hot_list)
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

/// The churn pool's record under the residency word `residency`, against
/// the plan it was taken from.
pub fn residency_host(
    residency: &str,
    pool: &model::placement::churn::ChurnPool,
    plan: &Plan<'_>,
) -> Record {
    Record::new(&RESIDENCY_HOST)
        .w("residency", residency)
        .u("pinned", pool.pinned)
        .u("churn_experts", pool.experts)
        .u("churn_bytes", pool.bytes)
        .u("headroom", plan.host.headroom_bytes)
        .u("headroom_after", pool.headroom_after(plan))
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
    Record::new(&RESIDENCY_PASS)
        .w("pass", kind.word())
        .u("boundary", r.boundary)
        .u("kept", r.kept)
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

/// A prompt call's pick record, of group `group`.
#[cfg(feature = "gpu")]
pub fn call_stream(group: usize, p: &CallPick) -> Record {
    Record::new(&CALL_STREAM)
        .u("group", group)
        .u("layer", p.layer)
        .u("admitted", p.admitted)
        .u("kept", p.kept)
        .u("bytes", p.bytes)
        .w("counts", format!("{:016x}", p.counts))
        .u("pick_us", p.pick_us)
        .u("backlog_us", p.backlog_us)
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
            GENERATE_GLM5NEXT,
            GENERATE_QWEN3MOE,
            GATE_DEEPSEEK41_PREFILL,
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
            .w("hot_list", "none")
            .line();
        assert_eq!(
            plan38,
            "plan place=a card=A6000 experts=card ctx_max=4096 host_experts=11735 card_experts=12841 hot_list=none"
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
        let lb = Record::new(&STAT_PROMPT38_LB)
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
            "stat prompt38 lb b=0 layer=3 cols=4096 slots=40960 experts=512 m_max=97 m_hot=0 \
             cols_hot=0 m_sq=3312000 wait_ms=21.53 union_ms=214.87 \
             serve_ms=236.71 enqueue_ms=0.94 card_front_ms=19.62 card_down_ms=1.84 \
             card_shadow_ms=0.71 card_upload_ms=1.79 card_back_ms=0.21"
        );
        let split = Record::new(&STAT_PROMPT38_SPLIT)
            .u("ubatches", 1)
            .u("layer_batches", 48)
            .f("prologue_ms", 18.4)
            .f("walk_ms", 13780.9)
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
            "stat prompt38 split ubatches=1 layer_batches=48 prologue_ms=18.4 walk_ms=13780.9 \
             union_ms=10313.8 wait_ms=1033.4 serve_ms=11364.1 enqueue_ms=2416.8 \
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
                "generate_qwen3moe",
                GENERATE_QWEN3MOE,
                include_str!("../../../tools/bloomery/schema/generate_qwen3moe.jsonl"),
            ),
            (
                "gate_deepseek41_prefill",
                GATE_DEEPSEEK41_PREFILL,
                include_str!("../../../tools/bloomery/schema/gate_deepseek41_prefill.jsonl"),
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
}
