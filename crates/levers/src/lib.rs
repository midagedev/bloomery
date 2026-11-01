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
//! or harness that owns it. A lever's row also says its [`Phase`]: fixed
//! when the model is loaded or the process starts, or changeable between
//! calls of one loaded model. A value the kind does not take is a refusal
//! that names the lever, the value and what it takes — never the default.

use std::ffi::{OsStr, OsString};
use std::fmt::{self, Write as _};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

mod registry;
use registry::REGISTRY;
pub use registry::{
    CARD_BUDGET, CARD_DONTNEED, CARD_MARGIN, CED, CHECK_FINITE, DRAFT, ENGRAM_HELPER, FORCE_IDS,
    GEN_SLOTS, GEN_SLOTS_MAX, HOST_LANES, HOST_LANES_DEFAULT, HOST_LOCK, HOST_POPULATE, HOST_ROOM,
    HOSTSTREAM, LANE_PREFETCH, LANE_PREFETCH_DEFAULT, MTP_DRAFT, MTP_HEAD_ROWS, MTP_WIDTH,
    MTP_WINDOWS, NEED_HOOK, NVTIER_BYTES, NVTIER_EVICT, NVTIER_MEM, NVTIER_READ, PIN_MAIN, PREFILL,
    PREFILL_GROUP, PREFILL_GROUP_DEFAULT, PREFILL_GROUP_MAX, QWEN3_KV, QWEN38_EXPERTS, R8,
    RESIDENCY, ROUTE_TRACE, SCREEN, SPIN, STEP_STATS, THREADS, TIER_LINK, TIER_NODES, TP_NODES,
    WARM_PROFILE,
};
pub use registry::{PREPARE_Q8, SCHEMAJUMP, SPLIT_WALK, XSTREAM};

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
    /// A [`CardBudget`] ([`card_budget_value`]): one byte count, or
    /// `CUDA<N>=<bytes>` per card, comma-separated, or `CUDA<N>=<n>x` shares
    /// per card.
    CardBytes,
    /// A non-empty path.
    Path,
    /// One of these words, or else a non-empty path: a word is never read
    /// as a path (a file of that name is named `./<word>`).
    PathOr(&'static [&'static str]),
    /// The path of a regular file that exists when the value is read.
    File,
    /// The absolute path of a regular file that exists when the value is
    /// read, which the process runs: a name with no directory would be
    /// searched in `PATH`, and a relative one means the working directory
    /// of the moment it runs, so neither is taken.
    Program,
    /// `<path>:<offset>`: the path of a regular file that exists when the
    /// value is read (holding no `:`), a `:` and a whole number.
    FileAt,
    /// What the name's owner takes: a name no reading here parses.
    Text,
    /// `off`, or the adaptive residency rule's `mid-p<P>-s<S>`
    /// ([`residency_word`]).
    Residency,
    /// Loopback socket addresses (`IP:port`), comma-separated, at least
    /// one, none twice: the tier page a node reaches is this host's memory,
    /// and the channel has no authentication.
    Addrs,
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
            Kind::CardBytes => card_budget_value(v)
                .map(Value::CardBudget)
                .map_err(|why| refused(why.as_deref())),
            Kind::Path if !v.is_empty() => Ok(Value::Path(PathBuf::from(v))),
            Kind::PathOr(words) => match words.iter().find(|&&w| w == v) {
                Some(&w) => Ok(Value::Word(w)),
                None if !v.is_empty() => Ok(Value::Path(PathBuf::from(v))),
                None => Err(refused(None)),
            },
            Kind::Program if !Path::new(v).is_absolute() && !v.is_empty() => Err(refused(Some(
                "a program is named by an absolute path, since a relative name is searched in \
                 PATH or in the working directory of the moment it runs",
            ))),
            Kind::File | Kind::Program if Path::new(v).is_file() => {
                Ok(Value::Path(PathBuf::from(v)))
            }
            Kind::File | Kind::Program if !v.is_empty() => {
                Err(refused(Some("no regular file is at that path")))
            }
            Kind::FileAt => {
                let (path, at) = v.rsplit_once(':').ok_or_else(|| refused(None))?;
                if path.contains(':') {
                    return Err(refused(Some("the path holds a ':'")));
                }
                if !Path::new(path).is_file() {
                    return Err(refused(Some("no regular file is at that path")));
                }
                match whole(at) {
                    Ok(at) => Ok(Value::PathAt(PathBuf::from(path), at)),
                    Err(NotWhole::Syntax) => Err(refused(None)),
                    Err(NotWhole::Overflow) => Err(refused(Some(OVERFLOW))),
                }
            }
            Kind::Residency => match residency_word(v) {
                Some(ResidencyWord::Off) => Ok(Value::Word("off")),
                // Parsed once at `main`: the word lives for the process, as
                // a listed word would.
                Some(ResidencyWord::Mid { .. }) => {
                    Ok(Value::Word(Box::leak(v.to_owned().into_boxed_str())))
                }
                None => Err(refused(None)),
            },
            Kind::Addrs => {
                let mut addrs: Vec<SocketAddr> = Vec::new();
                for a in v.split(',') {
                    let a: SocketAddr = a.parse().map_err(|_| refused(None))?;
                    if !tier_ip(a.ip()) {
                        return Err(refused(Some(
                            "a tier node needs a loopback, private or 198.18.0.0/15 IPv4 address",
                        )));
                    }
                    if addrs.contains(&a) {
                        return Err(refused(Some("an address is given twice")));
                    }
                    addrs.push(a);
                }
                Ok(Value::Addrs(addrs))
            }
            Kind::Path | Kind::File | Kind::Program | Kind::Text => Err(refused(None)),
        }
    }

    /// `v` as a value of this kind for a reading that checks it: a
    /// [`Kind::Text`] value is its owner's to check, taken as written
    /// (`None`); any other is [`Kind::parse`]'s.
    pub(crate) fn taken(self, v: &str) -> Result<Option<Value>, String> {
        match self {
            Kind::Text => Ok(None),
            kind => kind.parse(v).map(Some),
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
            Kind::CardBytes => format!(
                "bytes for every card, or CUDA<N>=<bytes> for each card named, comma-separated, \
                 each card once (N its CUDA ordinal, {DIGITS}; bytes a count, or a whole number \
                 of MiB or GiB with an M or G suffix)"
            ),
            Kind::Path => "a non-empty path".to_string(),
            Kind::PathOr(words) => format!("{}, or a non-empty path", words.join(", ")),
            Kind::File => "the path of an existing regular file".to_string(),
            Kind::Program => "the absolute path of an existing regular file".to_string(),
            Kind::FileAt => format!(
                "<path>:<offset>, the path of an existing regular file holding no ':' and a \
                 whole number ({DIGITS})"
            ),
            Kind::Text => "what its owner takes".to_string(),
            Kind::Addrs => {
                "loopback, private or 198.18.0.0/15 IPv4 IP:port socket addresses, comma-separated, at least one, none twice"
                    .to_string()
            }
            Kind::Residency => {
                format!(
                    "off, or mid-p<P>-s<S>, optionally ending in -unhosted, with P and S whole \
                     numbers ({DIGITS}), S at least 1"
                )
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
    /// victim, `spares` free slots a layer; `unhosted` the word ends in
    /// [`UNHOSTED_SUFFIX`]: it asks the load to keep the churn pool unhosted.
    Mid {
        pinned: usize,
        spares: usize,
        unhosted: bool,
    },
}

/// The suffix of a `mid` word that asks a load to keep its churn pool
/// unhosted: `mid-p<P>-s<S>-unhosted`.
pub const UNHOSTED_SUFFIX: &str = "-unhosted";

/// `v` in the residency lever's grammar, the one owner of it: `off`, or
/// `mid-p<P>-s<S>` with `P` and `S` whole numbers in base 10 with no sign or
/// leading zero and `S` at least 1, optionally followed by
/// [`UNHOSTED_SUFFIX`]; `None` for anything else.
#[must_use]
pub fn residency_word(v: &str) -> Option<ResidencyWord> {
    if v == "off" {
        return Some(ResidencyWord::Off);
    }
    let (v, unhosted) = match v.strip_suffix(UNHOSTED_SUFFIX) {
        Some(mid) => (mid, true),
        None => (v, false),
    };
    let (p, s) = v.strip_prefix("mid-p")?.split_once("-s")?;
    let num = |t: &str| usize::try_from(whole(t).ok()?).ok();
    match (num(p)?, num(s)?) {
        (pinned, spares) if spares >= 1 => Some(ResidencyWord::Mid {
            pinned,
            spares,
            unhosted,
        }),
        _ => None,
    }
}

/// What [`CARD_BUDGET`] asks a plan's cards for: a card plans with
/// `min(usable, budget)` usable bytes, or with the bytes that give it a share
/// of the experts the cards hold. The placement applies it per card
/// (`placement::card_budget`); this crate owns its spelling
/// ([`card_budget_value`]). [`CARD_MARGIN`] reads the same spelling for the
/// bytes a card keeps free.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum CardBudget {
    /// Unset: each card plans with its own usable bytes.
    #[default]
    Unset,
    /// One budget every card of the plan takes.
    Every(u64),
    /// A budget per card, by this process's CUDA ordinal (the `CUDA<N>`
    /// `--list-devices` prints), in the order given, each ordinal once; a
    /// card the list does not name plans with its own usable bytes.
    Per(Vec<(u32, u64)>),
    /// A share per card, by CUDA ordinal as [`CardBudget::Per`] names them,
    /// each ordinal once and at least one share above 0: the plan turns the
    /// shares into a budget per card from its own expert bytes
    /// (`placement::card_budget::split`), so the cards hold experts in this
    /// proportion. The shares are whole numbers of parts, `CUDA0=3x,CUDA1=1x`.
    Shares(Vec<(u32, u64)>),
}

impl CardBudget {
    /// Whether a budget is set, one, per card or by shares.
    #[must_use]
    pub fn is_set(&self) -> bool {
        *self != CardBudget::Unset
    }

    /// The one budget every card takes; `None` unset, per card or by shares.
    #[must_use]
    pub fn every(&self) -> Option<u64> {
        match self {
            CardBudget::Every(b) => Some(*b),
            CardBudget::Unset | CardBudget::Per(_) | CardBudget::Shares(_) => None,
        }
    }

    /// The `(ordinal, value)` entries of a per-card list, bytes or shares; `None` for
    /// unset and one value for every card.
    #[must_use]
    pub fn listed(&self) -> Option<&[(u32, u64)]> {
        match self {
            CardBudget::Per(list) | CardBudget::Shares(list) => Some(list),
            CardBudget::Unset | CardBudget::Every(_) => None,
        }
    }
}

/// A plan's one budget, or none: what a caller that sizes every card alike
/// hands over.
impl From<Option<u64>> for CardBudget {
    fn from(bytes: Option<u64>) -> CardBudget {
        bytes.map_or(CardBudget::Unset, CardBudget::Every)
    }
}

/// The value as the lever spells it (`none` unset): what a record prints.
impl fmt::Display for CardBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CardBudget::Unset => write!(f, "none"),
            CardBudget::Every(b) => write!(f, "{b}"),
            CardBudget::Per(per) => {
                for (i, (o, b)) in per.iter().enumerate() {
                    let sep = if i == 0 { "" } else { "," };
                    write!(f, "{sep}CUDA{o}={b}")?;
                }
                Ok(())
            }
            CardBudget::Shares(shares) => {
                for (i, (o, w)) in shares.iter().enumerate() {
                    let sep = if i == 0 { "" } else { "," };
                    write!(f, "{sep}CUDA{o}={w}x")?;
                }
                Ok(())
            }
        }
    }
}

/// `v` in [`CARD_BUDGET`]'s grammar, the one owner of it: one byte count
/// ([`parse_bytes`]) for every card, or `CUDA<N>=<bytes>` entries separated
/// by `,`, `CUDA` in any ASCII case and `N` a whole number with no sign or
/// leading zero, each ordinal once; or the same entries as shares,
/// `CUDA<N>=<n>x` with `n` a whole number, every entry of a list one kind and
/// one share above 0 at least. A refusal's reason names the entry where the
/// grammar alone does not say why; `None` where it does.
pub fn card_budget_value(v: &str) -> Result<CardBudget, Option<String>> {
    if !v.contains('=') {
        return match parse_bytes(v) {
            Ok(b) => Ok(CardBudget::Every(b)),
            Err(e @ BytesError::Overflow(_)) => Err(Some(e.reason().to_string())),
            Err(BytesError::NotBytes(_)) => Err(None),
        };
    }
    let mut per: Vec<(u32, u64)> = Vec::new();
    let mut shared: Option<bool> = None;
    for entry in v.split(',') {
        let (card, bytes) = entry
            .split_once('=')
            .ok_or_else(|| Some(format!("{entry:?} is no CUDA<N>=<bytes> entry")))?;
        let ordinal = card
            .get(..4)
            .filter(|p| p.eq_ignore_ascii_case("cuda"))
            .and_then(|_| whole(&card[4..]).ok())
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| Some(format!("{card:?} names no device as CUDA<N>")))?;
        let share = bytes.ends_with(['x', 'X']);
        if *shared.get_or_insert(share) != share {
            return Err(Some(format!(
                "CUDA{ordinal}: a list is all bytes or all shares (`<n>x`), not both"
            )));
        }
        let n = if share {
            let parts = &bytes[..bytes.len() - 1];
            whole(parts).map_err(|e| match e {
                NotWhole::Syntax => Some(format!(
                    "{bytes:?} for CUDA{ordinal} is no share (a whole number, then x)"
                )),
                NotWhole::Overflow => Some(format!("CUDA{ordinal}: {OVERFLOW}")),
            })?
        } else {
            parse_bytes(bytes).map_err(|e| match e {
                BytesError::NotBytes(_) => Some(format!("{bytes:?} for CUDA{ordinal} is no bytes")),
                BytesError::Overflow(_) => Some(format!("CUDA{ordinal}: {}", e.reason())),
            })?
        };
        if per.iter().any(|&(o, _)| o == ordinal) {
            return Err(Some(format!("CUDA{ordinal} is given twice")));
        }
        per.push((ordinal, n));
    }
    if shared != Some(true) {
        return Ok(CardBudget::Per(per));
    }
    match per.iter().try_fold(0u64, |sum, &(_, w)| sum.checked_add(w)) {
        None => Err(Some(format!("the shares sum to {OVERFLOW}"))),
        Some(0) => Err(Some(
            "every share is 0: a split needs one above 0".to_string(),
        )),
        Some(_) => Ok(CardBudget::Shares(per)),
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
    /// A [`Kind::CardBytes`] budget, never [`CardBudget::Unset`].
    CardBudget(CardBudget),
    /// A [`Kind::Path`], [`Kind::PathOr`], [`Kind::File`] or [`Kind::Program`] path.
    Path(PathBuf),
    /// A [`Kind::FileAt`] path and offset.
    PathAt(PathBuf, u64),
    /// A [`Kind::Addrs`] list, in the order given.
    Addrs(Vec<SocketAddr>),
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

/// When a lever's effect can change: with the load (or the process start)
/// that reads it, or between calls of one loaded model.
///
/// The definition: a lever is [`Phase::Runtime`] when a binary can change
/// its effect between calls of one loaded model, with no reload, through a
/// named API — a setter function or a per-call argument the engine takes —
/// and the environment value is only the initial one. Every other lever row
/// is [`Phase::Load`]: its effect is fixed when the model is loaded or the
/// process starts — it shapes the plan, the bytes, the buffers, the captured
/// graphs or the process's pools — or it is read once at `main` and handed
/// down with no way to change it later; a diagnostic read once at `main`
/// (`BLOOMERY_STEP_STATS`) is Load.
///
/// A lever that several model families parse is Runtime only when every
/// family that parses it has the setter: `BLOOMERY_PREFILL_GROUP` is Load
/// today — the `set_prefill_group` of GLM-5.3 and MiMo-V2.6-Flash beside a
/// V4.1 load that fixes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    /// The effect is fixed when the model is loaded or the process starts.
    Load,
    /// Changeable between calls of one loaded model, no reload, through the
    /// named API `setter`; the environment value is the initial one.
    ///
    /// No row constructs it today: the registry holds no Runtime lever, so
    /// the dead-code lint is expected here until the first one lands.
    #[expect(dead_code)]
    Runtime { setter: Setter },
}

/// The named API a [`Phase::Runtime`] lever changes through: the function a
/// binary calls between calls of one loaded model. `tools/check-levers.sh`'s
/// fifth hold checks the file is in the tree and defines `fn <item>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Setter {
    /// The file the function is defined in, from the repository root.
    pub(crate) file: &'static str,
    /// The function's name — a free function or a method — as `fn <item>`
    /// is defined in that file.
    pub(crate) item: &'static str,
}

impl Phase {
    /// The `--levers` table's word for the phase.
    pub(crate) fn word(self) -> &'static str {
        match self {
            Phase::Load => "load",
            Phase::Runtime { .. } => "runtime",
        }
    }
}

/// Who reads a name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Site {
    /// This crate parses it: at a binary's `main` ([`at_main`]), or where the
    /// worker pool is built ([`pool_levers`]). `left` names the files that
    /// still read it in place besides.
    Parsed {
        left: &'static [InPlace],
        phase: Phase,
    },
    /// Read in place in the files of `at` only, by the binaries `by` says;
    /// a binary's reading here checks its value by its kind, read or not,
    /// and names it as not applied where the binary does not read it.
    Direct {
        at: &'static [InPlace],
        phase: Phase,
        by: Readers,
    },
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

/// Which binaries read a [`Site::Direct`] lever.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Readers {
    /// Every binary: the files read it below any binary's `main`, wherever
    /// the host tier runs, as the pool reads its two; no `at_main` list names
    /// it, and none refuses it.
    Every,
    /// The binaries whose `at_main` list names it, its files on their path;
    /// every other binary takes it, set, as not applied and says so: one
    /// environment may carry levers for several models.
    Named,
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
            Site::Parsed { left: [], .. } => "parsed".to_string(),
            Site::Parsed { left, .. } => format!("parsed; in place in {}", places(left)),
            Site::Direct {
                at,
                by: Readers::Every,
                ..
            } => format!("in place in {}, by every binary", places(at)),
            Site::Direct { at, .. } => format!("in place in {}", places(at)),
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
    /// A value handed over in place of the environment's ([`override_with`])
    /// that no override can carry: what stops it.
    Override(&'static str),
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
            Why::Override(why) => write!(f, "{why}"),
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

/// One lever as a reading found it: a Parsed lever, or a Direct one a
/// binary's reading checks.
#[derive(Clone, Debug)]
struct Entry {
    row: &'static LeverSpec,
    /// Whether the reading's binary acts on it. A Parsed lever it does not
    /// act on is unset (set, it was refused) and holds the default; a Direct
    /// one holds what is set, which it does not apply.
    acts: bool,
    /// The text as set; `None` unset.
    set: Option<String>,
    /// The value: as set, else the default; `None` unset when unset is a
    /// state of its own ([`Unset::Means`]), and set for a [`Kind::Text`]
    /// value, which its owner checks.
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
            .taken(text)
            .map_err(|expected| refuse(text.to_string(), expected))?;
        Ok(Entry {
            row,
            acts,
            set: Some(text.to_string()),
            value,
        })
    }
}

/// Whose reading it is, which fixes what it refuses besides a value a
/// lever's kind does not take and a retired name that is set.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Scope<'a> {
    /// A binary's `main` ([`at_main`]): a Parsed lever set that is not in
    /// `acts_on` (the pool's two always are), and a `BLOOMERY_*` name no row
    /// names; a Direct lever is checked by its kind, as a Parsed one is,
    /// and a [`Readers::Named`] one that is not in `acts_on` is taken as not
    /// applied ([`Reading::unapplied`]).
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
    /// The host's room and the NVMe expert tier's four levers
    /// ([`nvtier_levers`]), read where they act — the host's reading, the
    /// placement dial and the tier's build, all below every binary's `main`
    /// — like the pool's two.
    NvTier,
    /// The load's need hook ([`NEED_HOOK`]) and the lock it states
    /// ([`HOST_LOCK`]), read where a plan prints its need record and a placed
    /// load consults the hook ([`need_levers`]), below every binary's `main`.
    /// Only the hook is acted on by every binary; the lock stays on each
    /// binary's `acts_on` list.
    Need,
    /// How the tier page crosses ([`tier_link`]), read where the host tier
    /// takes its tier cards, below every binary's `main`.
    TierLink,
    /// The expert tiers' nodes ([`tier_nodes`]), read where a load takes its
    /// tier cards, below every binary's `main`.
    TierNodes,
    /// The cards' budget ([`card_budget`]), read where a placement meets the
    /// census, below every binary's `main`, which still refuses it in a
    /// binary that does not act on it; and the cards' margin
    /// ([`card_margin`]), read there too.
    CardBudget,
}

/// What one reading found: the levers it took and every variable it refused.
#[derive(Debug)]
pub(crate) struct Reading {
    entries: Vec<Entry>,
    refused: Vec<Refusal>,
}

/// The levers of `scope` over `env`, every name once — the last of a name
/// given twice — and every refusal at once.
pub(crate) fn read(env: &[(OsString, OsString)], scope: Scope<'_>) -> Result<Levers, LeverError> {
    reading(env, scope).levers()
}

/// [`read`]'s reading, its refusals beside what it took: what `--levers`
/// prints ([`Reading::table`]).
pub(crate) fn reading(env: &[(OsString, OsString)], scope: Scope<'_>) -> Reading {
    if let Scope::Main { acts_on, .. } = scope
        && let Some(name) = not_named(acts_on)
    {
        panic!("{name} is not a lever a binary names among those it acts on");
    }
    let get = |name: &str| {
        env.iter()
            .rev()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_os_str())
    };
    let pool = |name: &str| name == THREADS || name == SPIN;
    let nvtier = |name: &str| {
        name == HOST_ROOM
            || name == NVTIER_BYTES
            || name == NVTIER_READ
            || name == NVTIER_MEM
            || name == NVTIER_EVICT
    };
    let link = |name: &str| name == TIER_LINK;
    let nodes = |name: &str| name == TIER_NODES;
    let hook = |name: &str| name == NEED_HOOK;
    let margin = |name: &str| name == CARD_MARGIN;
    let mut entries = Vec::new();
    let mut refused = Vec::new();
    for row in REGISTRY {
        let acts = match (row.site, scope) {
            (Site::Parsed { .. }, Scope::Main { acts_on, .. }) => {
                pool(row.name)
                    || nvtier(row.name)
                    || link(row.name)
                    || nodes(row.name)
                    || hook(row.name)
                    || margin(row.name)
                    || acts_on.contains(&row.name)
            }
            (Site::Direct { by, .. }, Scope::Main { acts_on, .. }) => {
                by == Readers::Every || acts_on.contains(&row.name)
            }
            (Site::Parsed { .. }, Scope::Every) => true,
            (Site::Parsed { .. }, Scope::Pool) if pool(row.name) => true,
            (Site::Parsed { .. }, Scope::NvTier) if nvtier(row.name) => true,
            (Site::Parsed { .. }, Scope::Need) if hook(row.name) || row.name == HOST_LOCK => true,
            (Site::Parsed { .. }, Scope::TierLink) if link(row.name) => true,
            (Site::Parsed { .. }, Scope::TierNodes) if nodes(row.name) || link(row.name) => true,
            (Site::Parsed { .. }, Scope::CardBudget)
                if row.name == CARD_BUDGET || margin(row.name) =>
            {
                true
            }
            (Site::Retired { why, .. }, _) => {
                if let Some(v) = get(row.name) {
                    refused.push(Refusal {
                        name: row.name.to_string(),
                        value: lossy(v),
                        why: Why::Retired(why),
                    });
                }
                continue;
            }
            _ => continue,
        };
        let parsed = matches!(row.site, Site::Parsed { .. });
        match (get(row.name), scope) {
            (Some(v), Scope::Main { bin, .. }) if !acts && parsed => refused.push(Refusal {
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
    let rdma = entries
        .iter()
        .any(|e| e.row.name == TIER_LINK && e.value == Some(Value::Word("rdma")));
    let nodes_of = |name: &str| {
        entries
            .iter()
            .find(|e| e.row.name == name)
            .and_then(|e| match &e.value {
                Some(Value::Addrs(addrs)) => Some((e, addrs)),
                _ => None,
            })
    };
    for name in [TIER_NODES, TP_NODES] {
        if let Some((entry, addrs)) = nodes_of(name)
            && addrs.iter().any(|a| !a.ip().is_loopback())
            && !rdma
        {
            refused.push(Refusal {
                name: name.to_string(),
                value: entry
                    .set
                    .clone()
                    .expect("node addresses came from a set lever"),
                why: Why::Value(format!("a node off this host requires {TIER_LINK}=rdma")),
            });
        }
    }
    if let (Some(_), Some((tp, _))) = (nodes_of(TIER_NODES), nodes_of(TP_NODES)) {
        refused.push(Refusal {
            name: TP_NODES.to_string(),
            value: tp
                .set
                .clone()
                .expect("node addresses came from a set lever"),
            why: Why::Value(format!(
                "a tensor-parallel run's nodes are its ranks; {TIER_NODES} names expert tier \
                 nodes, and the two are not set together"
            )),
        });
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
    Reading { entries, refused }
}

/// The first of `names` that is no lever a binary names among those it acts
/// on: a Parsed lever, or a [`Readers::Named`] Direct one.
fn not_named<'a>(names: &[&'a str]) -> Option<&'a str> {
    names.iter().copied().find(|&name| {
        !spec(name).is_some_and(|r| {
            matches!(
                r.site,
                Site::Parsed { .. }
                    | Site::Direct {
                        by: Readers::Named,
                        ..
                    }
            )
        })
    })
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
        let env = process_env();
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

    fn path(&self, name: &str) -> Option<&Path> {
        match &self.entry(name).value {
            None => None,
            Some(Value::Path(p)) => Some(p),
            v => panic!("{name} holds {v:?}, not a path"),
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

    /// `BLOOMERY_PREFILL_GROUP`: batches a V4.1, GLM-5.3 or MiMo-V2.6-Flash
    /// prompt group holds, 1 to [`PREFILL_GROUP_MAX`].
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

    /// `BLOOMERY_CARD_BUDGET`: the cards' budget, one or per card;
    /// [`CardBudget::Unset`] unset.
    #[must_use]
    pub fn card_budget(&self) -> CardBudget {
        match &self.entry(CARD_BUDGET).value {
            None => CardBudget::Unset,
            Some(Value::CardBudget(b)) => b.clone(),
            v => panic!("{CARD_BUDGET} holds {v:?}, not a card budget"),
        }
    }

    /// `BLOOMERY_CARD_MARGIN`: the bytes each card keeps free, one value or
    /// one per card; [`CardBudget::Unset`] unset, each card's own margin.
    #[must_use]
    pub fn card_margin(&self) -> CardBudget {
        match &self.entry(CARD_MARGIN).value {
            None => CardBudget::Unset,
            Some(Value::CardBudget(b)) => b.clone(),
            v => panic!("{CARD_MARGIN} holds {v:?}, not a card byte list"),
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
        self.path(MTP_DRAFT)
    }

    /// `BLOOMERY_MTP_WINDOWS`: every drafted window's proposal, probabilities
    /// and kept count are printed.
    #[must_use]
    pub fn mtp_windows(&self) -> bool {
        self.flag(MTP_WINDOWS)
    }

    /// `BLOOMERY_MTP_WIDTH` as set, `cost` or `fixed`; `None` unset, which a
    /// run that drafts reads as `cost` (`runtime::width::Mode::of`).
    #[must_use]
    pub fn mtp_width(&self) -> Option<&'static str> {
        self.word(MTP_WIDTH)
    }

    /// `BLOOMERY_ROUTE_TRACE`: the route trace's new directory; `None` unset
    /// (no trace).
    #[must_use]
    pub fn route_trace(&self) -> Option<&Path> {
        self.path(ROUTE_TRACE)
    }

    /// `BLOOMERY_WARM_PROFILE`: the residency's usage profile; `None` unset.
    #[must_use]
    pub fn warm_profile(&self) -> Option<&Path> {
        self.path(WARM_PROFILE)
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
    /// `None` unset, which a load reads as `f16` (`bloomery_gpu::kv_format`).
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

    /// `BLOOMERY_XSTREAM`: what a Qwen3.8 prompt call moves — `off`, `admit`
    /// (the residency's pick) or `split` (the pick, then the expert stream);
    /// `None` when unset, which the binary resolves.
    #[must_use]
    pub fn xstream(&self) -> Option<&'static str> {
        self.word(XSTREAM)
    }

    /// `BLOOMERY_SPLIT_WALK`: a Qwen3.8 prompt walks a ubatch the width rule
    /// finds wide enough as two units; `None` when unset, which the load
    /// resolves.
    #[must_use]
    pub fn split_walk(&self) -> Option<bool> {
        match self.entry(SPLIT_WALK).value {
            None => None,
            Some(Value::Flag(b)) => Some(b),
            ref v => panic!("{SPLIT_WALK} holds {v:?}, not a flag"),
        }
    }

    /// `BLOOMERY_LANE_PREFETCH`: the host union's row-lane packs prefetch
    /// each group's successor.
    #[must_use]
    pub fn lane_prefetch(&self) -> bool {
        self.flag(LANE_PREFETCH)
    }

    /// `BLOOMERY_HOST_LANES`: how the host tier's row dispatch lays its units
    /// over the pool.
    #[must_use]
    pub fn host_lanes(&self) -> HostLanes {
        match self.word(HOST_LANES) {
            Some("flat") => HostLanes::Flat,
            Some("ccd") => HostLanes::Ccd,
            w => panic!("{HOST_LANES} holds {w:?}, not flat or ccd"),
        }
    }

    /// `BLOOMERY_NVTIER_MEM`: the memory the NVMe tier's arena is.
    #[must_use]
    pub fn nvtier_mem(&self) -> NvTierMem {
        match self.word(NVTIER_MEM) {
            Some("auto") => NvTierMem::Auto,
            Some("anon") => NvTierMem::Anon,
            Some("pinned") => NvTierMem::Pinned,
            w => panic!("{NVTIER_MEM} holds {w:?}, not auto, anon or pinned"),
        }
    }

    /// `BLOOMERY_NVTIER_EVICT`: the victim rule of the NVMe tier's arena.
    #[must_use]
    pub fn nvtier_evict(&self) -> NvTierEvict {
        match self.word(NVTIER_EVICT) {
            Some("tick") => NvTierEvict::Tick,
            Some("heat") => NvTierEvict::Heat,
            w => panic!("{NVTIER_EVICT} holds {w:?}, not tick or heat"),
        }
    }

    /// `BLOOMERY_TP_NODES`: the node of each rank past rank 0, in rank order;
    /// empty when unset.
    #[must_use]
    pub fn tp_nodes(&self) -> &[SocketAddr] {
        match &self.entry(TP_NODES).value {
            None => &[],
            Some(Value::Addrs(a)) => a,
            v => panic!("{TP_NODES} holds {v:?}, not socket addresses"),
        }
    }

    /// `BLOOMERY_GEN_SLOTS`: the streams `generate_qwen3moe` decodes in one
    /// pass, 1 to [`GEN_SLOTS_MAX`].
    #[must_use]
    pub fn gen_slots(&self) -> usize {
        let n = Levers::defaulted(GEN_SLOTS, self.count(GEN_SLOTS));
        Levers::usize_of(GEN_SLOTS, n)
    }

    /// `BLOOMERY_FORCE_IDS`: the file of ids (one a line), which existed
    /// when the reading ran, and the offset its ids are taken from; `None`
    /// unset (each row's argmax).
    #[must_use]
    pub fn force_ids(&self) -> Option<(&Path, u64)> {
        match &self.entry(FORCE_IDS).value {
            None => None,
            Some(Value::PathAt(p, at)) => Some((p, *at)),
            v => panic!("{FORCE_IDS} holds {v:?}, not a path and an offset"),
        }
    }

    /// `BLOOMERY_SCREEN`: a serving seat draws its start screen on a
    /// terminal, and `d` opens its live view.
    #[must_use]
    pub fn screen(&self) -> bool {
        self.flag(SCREEN)
    }

    /// `BLOOMERY_SCHEMAJUMP`: a server runs a request that follows a program
    /// on the jump arm (the default), or on the step arm alone.
    #[must_use]
    pub fn schema_jump(&self) -> bool {
        self.flag(SCHEMAJUMP)
    }

    /// `BLOOMERY_PREPARE_Q8`: `prepare` writes the BF16 or F32 matrices its map
    /// reads as Q8_0 in Q8_0, not as stored.
    #[must_use]
    pub fn prepare_q8(&self) -> bool {
        self.flag(PREPARE_Q8)
    }

    /// The host tier's load settings: [`HOST_POPULATE`], [`HOST_LOCK`],
    /// [`CARD_DONTNEED`], [`R8`], and whether [`ROUTE_TRACE`] is set.
    #[must_use]
    pub fn host(&self) -> HostCfg {
        HostCfg {
            populate: self.flag(HOST_POPULATE),
            lock: self.flag(HOST_LOCK),
            card_dontneed: self.flag(CARD_DONTNEED),
            r8: self.flag(R8),
            route_trace: self.route_trace().is_some(),
        }
    }
}

impl Reading {
    /// Every lever row of [`REGISTRY`], one line each, with four cells and who
    /// reads it. The phase: `load`, `runtime`, or `-` for a retired row,
    /// which states none. Whether this run applies it: `yes`, read at the
    /// value shown; `no`, its binary does not read it ([`Reading::unapplied`]
    /// says why when it is set); `refused`, set and refused, the reading's
    /// refusal saying why. The value: as set, or the default, or `-` for an
    /// unset lever its binary does not read. What `--levers` prints
    /// ([`at_main`]); a name that is no lever has no line.
    pub(crate) fn table(&self) -> String {
        let mut out = String::new();
        for row in REGISTRY.iter().filter(|r| r.is_lever()) {
            let refusal = self.refused.iter().find(|f| f.name == row.name);
            let entry = self.entries.iter().find(|e| std::ptr::eq(e.row, row));
            let (applies, value) = match (refusal, entry) {
                (Some(f), _) => ("refused", format!("set {}", f.value)),
                (
                    None,
                    Some(Entry {
                        acts: false,
                        set: Some(text),
                        ..
                    }),
                ) => ("no", format!("set {text}")),
                (None, Some(Entry { acts: false, .. }) | None) => ("no", "-".to_string()),
                (
                    None,
                    Some(Entry {
                        set: Some(text), ..
                    }),
                ) => ("yes", format!("set {text}")),
                (None, Some(Entry { set: None, .. })) => match row.default {
                    Unset::Is(d) => ("yes", format!("default {d}")),
                    Unset::Means(m) => ("yes", format!("unset: {m}")),
                },
            };
            // A retired row is no lever to set; the phase cell is `-`.
            let phase = match row.site {
                Site::Parsed { phase, .. } | Site::Direct { phase, .. } => phase.word(),
                Site::Retired { .. } | Site::Env { .. } => "-",
            };
            let _ = writeln!(
                out,
                "{:<26} {:<9} {:<8} {:<7} {:<32} {}",
                row.name,
                row.class.describe(),
                phase,
                applies,
                value,
                row.site.describe()
            );
        }
        out
    }

    /// A line for each Direct lever set that the binary `bin` does not read:
    /// it runs, and the value does nothing there. What a start prints on
    /// stderr ([`at_main`]).
    pub(crate) fn unapplied(&self, bin: &str) -> Vec<String> {
        self.entries
            .iter()
            .filter(|e| !e.acts)
            .filter_map(|e| {
                let set = e.set.as_deref()?;
                Some(format!(
                    "{}={set:?} is not applied: this binary ({bin}) does not read it; it is read {}",
                    e.row.name,
                    e.row.site.describe()
                ))
            })
            .collect()
    }

    /// The levers, or every refusal at once ([`read`]).
    fn levers(self) -> Result<Levers, LeverError> {
        if self.refused.is_empty() {
            Ok(Levers {
                entries: self.entries,
            })
        } else {
            Err(LeverError {
                refused: self.refused,
            })
        }
    }
}

/// What a Qwen3.8 plan runs by with [`QWEN38_EXPERTS`] unset: the card
/// experts.
pub const QWEN38_EXPERTS_UNSET: &str = "card";

/// The residency word a serving placement runs by with [`RESIDENCY`] unset:
/// no seed expert pinned (the pin is the host-room fallback's, not the
/// default's), one spare a layer.
pub const RESIDENCY_SERVING: &str = "mid-p0-s1";

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
    /// Unset, the plan's card slots moved the word's pinned experts
    /// (`from`) to `to`, keeping its spares: `fewest` the plan's fewest card
    /// experts a layer.
    Shrunk {
        fewest: usize,
        from: usize,
        to: usize,
    },
    /// Unset, the plan holds no card expert.
    NoCardExperts,
    /// Unset, the plan holds no host expert: the machine would have none to
    /// move onto the card, and its spares would only displace card experts.
    NoHostExperts,
    /// Unset, the plan's fewest card experts a layer (`fewest`) leave no
    /// room for the word's P or half of them.
    NoRoom { fewest: usize },
    /// The word asks for its pool unhosted ([`UNHOSTED_SUFFIX`]): it keeps
    /// its P and the load does not host the churn pool there
    /// ([`PoolUnhosted`]). No unset resolution gives it.
    Unhosted(PoolUnhosted),
    /// Unset, on a load that hosts its pool: the word's pinned experts
    /// raised from `from` to `to`, the first count whose churn pool fits the
    /// host ([`Room::MovedForPool`]).
    PoolShrunk { from: usize, to: usize },
    /// Unset, on a load that hosts its pool: no count the card slots hold
    /// has a churn pool that fits the host ([`PoolUnhostable`]).
    PoolUnhostable(PoolUnhostable),
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
            ResidencyWhy::Shrunk { .. } => "shrunk",
            ResidencyWhy::NoCardExperts => "no_card_experts",
            ResidencyWhy::NoHostExperts => "no_host_experts",
            ResidencyWhy::NoRoom { .. } => "no_room",
            ResidencyWhy::Unhosted(_) => "pool_unhosted",
            ResidencyWhy::PoolShrunk { .. } => "pool_shrunk",
            ResidencyWhy::PoolUnhostable(_) => "pool_unhostable",
        }
    }

    /// The plan-side detail of an unset word the plan moved or refused, one
    /// line beside the `residency lever` record; `None` where the why word
    /// says it all.
    #[must_use]
    pub fn detail(self) -> Option<String> {
        match self {
            ResidencyWhy::Shrunk { fewest, from, to } => Some(format!(
                "unset: p{from}→p{to} (card: fewest {fewest}, half)"
            )),
            ResidencyWhy::NoCardExperts => {
                Some("unset: the plan holds no routed expert on the card".to_string())
            }
            ResidencyWhy::NoHostExperts => {
                Some("unset: the plan holds no routed expert on the host".to_string())
            }
            ResidencyWhy::NoRoom { fewest } => Some(format!(
                "unset: the plan's fewest card experts a layer ({fewest}) leave no room for the \
                 word's P or half of them"
            )),
            ResidencyWhy::Unhosted(pool) => Some(pool.to_string()),
            ResidencyWhy::PoolShrunk { from, to } => Some(format!(
                "unset: p{from}→p{to} (host: the churn pool did not fit, raised)"
            )),
            ResidencyWhy::PoolUnhostable(pool) => Some(pool.to_string()),
            ResidencyWhy::Set
            | ResidencyWhy::Place
            | ResidencyWhy::FixedPlace
            | ResidencyWhy::CheckFinite
            | ResidencyWhy::RouteTrace
            | ResidencyWhy::PrefillSteps => None,
        }
    }
}

impl From<RoomShort> for ResidencyWhy {
    fn from(short: RoomShort) -> ResidencyWhy {
        match short {
            RoomShort::NoCardExperts => ResidencyWhy::NoCardExperts,
            RoomShort::NoRoom { fewest } => ResidencyWhy::NoRoom { fewest },
            RoomShort::PoolUnhostable(pool) => ResidencyWhy::PoolUnhostable(pool),
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

impl ResidencyPick {
    /// Whether the load hosts the word's churn pool: unhosted exactly where
    /// the plan's room left the pool out ([`ResidencyWhy::Unhosted`]), the
    /// one place a pick's hosting is derived.
    #[must_use]
    pub fn pool(&self) -> ChurnHosting {
        hosting(matches!(self.why, ResidencyWhy::Unhosted(_)))
    }
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

/// The floor a family's word the plan moves its pinned count to keeps: a
/// move below it runs `off` instead. The floor is the sitting of
/// `docs/cards/v41shrink3090-ab.card`.
pub const RESIDENCY_MIN_P: usize = 1;

/// Whether a load under a `mid` word holds its churn pool in the host set it
/// reads in and locks, or only knows it: the one decision a word's
/// resolution carries to the load, which never derives it again.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChurnHosting {
    /// The pool is read in and locked with the plan's host segments, and the
    /// plan's host headroom takes its bytes.
    #[default]
    Hosted,
    /// The pool is known and not read in: its victims are served from the
    /// file's mapping, read back at a page fault's cost and counted.
    Unhosted,
}

/// Whether a body's load carries a churn pool it does not host: the body
/// declares it (`bloomery_gpu::model::ChainBody::UNHOSTED_POOL`) and a word
/// that asks for it ([`UNHOSTED_SUFFIX`]) reads it ([`room_for`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnhostedPool {
    /// The load takes [`ChurnHosting::Unhosted`] to its residency spec: a
    /// word that asks keeps its P with the pool unhosted ([`Room::Unhosted`]).
    /// An unset word asks for nothing and climbs as on any other body.
    Carried,
    /// The load hosts every churn pool it runs: a word that asks is refused
    /// by name ([`UnhostedPool::refuse`]), and an unset word whose pool fits
    /// neither host budget runs at the first P above it whose pool fits
    /// ([`Room::MovedForPool`]), `off` where none up to the card ceiling
    /// does ([`RoomShort::PoolUnhostable`]).
    NotCarried,
}

impl UnhostedPool {
    /// Whether a load takes its churn pool unhosted: the word asks for it
    /// (`asks`, [`UNHOSTED_SUFFIX`]) and the body carries one. An unset word
    /// asks for nothing.
    #[must_use]
    pub fn takes(self, asks: bool) -> bool {
        asks && self == UnhostedPool::Carried
    }

    /// The refusal of a word that asks for its churn pool unhosted
    /// ([`UNHOSTED_SUFFIX`]) on a load whose body hosts every pool it runs
    /// ([`unhosted_refusal`]); `None` where the body carries it.
    #[must_use]
    pub fn refuse(self, word: &str) -> Option<String> {
        match self {
            UnhostedPool::Carried => None,
            UnhostedPool::NotCarried => Some(unhosted_refusal(word)),
        }
    }
}

/// The refusal text of `word`, which asks for its churn pool unhosted, on a
/// load that hosts every churn pool it runs.
#[must_use]
pub fn unhosted_refusal(word: &str) -> String {
    format!(
        "BLOOMERY_RESIDENCY={word}: {UNHOSTED_SUFFIX} asks to keep the churn pool unhosted, \
         and this load hosts every churn pool it runs"
    )
}

/// The families' picks' one rule from their why: unhosted where the why is
/// the plan's room leaving the pool out ([`PoolUnhosted`]).
fn hosting(unhosted: bool) -> ChurnHosting {
    if unhosted {
        ChurnHosting::Unhosted
    } else {
        ChurnHosting::Hosted
    }
}

/// A word that asks for its churn pool unhosted ([`UNHOSTED_SUFFIX`]) on a
/// load that carries it: the word keeps its P and the load does not host the
/// pool ([`ChurnHosting::Unhosted`]). `needs` the pool's bytes at P,
/// `headroom` the plan's host headroom, `mem_left` what the host's room
/// leaves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolUnhosted {
    pub needs: u64,
    pub headroom: i128,
    pub mem_left: i128,
}

impl fmt::Display for PoolUnhosted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let PoolUnhosted {
            needs,
            headroom,
            mem_left,
        } = *self;
        write!(
            f,
            "the churn pool needs {needs} B, the plan leaves {headroom} B and the host's room \
             {mem_left} B: the word asks for the pool unhosted, its victims are served from the \
             file"
        )
    }
}

/// An unset word on a load that hosts every churn pool it runs
/// ([`UnhostedPool::NotCarried`]) whose pool fits neither budget at any P up
/// to what the card slots hold: `needs` the pool's bytes at the last P
/// probed, `headroom` the plan's host headroom, `mem_left` what the host's
/// room leaves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolUnhostable {
    pub needs: u64,
    pub headroom: i128,
    pub mem_left: i128,
}

impl fmt::Display for PoolUnhostable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let PoolUnhostable {
            needs,
            headroom,
            mem_left,
        } = *self;
        write!(
            f,
            "unset: the churn pool needs {needs} B at the highest P the card slots hold, the plan \
             leaves {headroom} B and the host's room {mem_left} B, and this load hosts its pool"
        )
    }
}

/// Why a family's residency word cannot run on a plan ([`Room::Short`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoomShort {
    /// No layer holds a card expert.
    NoCardExperts,
    /// The plan's fewest card experts a layer (`fewest`) leave no room for
    /// the word's P or half of them, their spares and one that moves.
    NoRoom { fewest: usize },
    /// The load hosts its pool and no P the card slots hold has a pool that
    /// fits ([`PoolUnhostable`]).
    PoolUnhostable(PoolUnhostable),
}

/// What a plan's room left of a family's target residency word
/// ([`room_for`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Room {
    /// The word runs as it is: the plan's card slots hold its pinned count,
    /// its spares and one that moves, and its host takes the churn pool
    /// there.
    AsIs,
    /// The word runs at a moved pinned count, keeping its spares: `from` the
    /// family's target, `to` what the plan's card slots left, on a plan
    /// whose fewest card experts a layer is `fewest`; its host takes the
    /// churn pool there.
    Moved {
        fewest: usize,
        from: usize,
        to: usize,
    },
    /// The word runs at `to`, the first pinned count above the P the card
    /// slots left whose churn pool fits both host budgets, keeping its
    /// spares; `from` the family's target. A load that hosts every pool it
    /// runs ([`UnhostedPool::NotCarried`]) only.
    MovedForPool { from: usize, to: usize },
    /// The word runs at `pinned`, the P the card slots left, and its host
    /// does not take the churn pool there ([`PoolUnhosted`]), whether or not
    /// the pool would fit: the word asked for it ([`UNHOSTED_SUFFIX`]). A
    /// load that carries an unhosted pool ([`UnhostedPool::Carried`]) only.
    Unhosted { pinned: usize, pool: PoolUnhosted },
    /// The word cannot run: `off`, and why ([`RoomShort`]).
    Short(RoomShort),
}

/// The pinned count a word of `target` pinned experts and `spares` free slots
/// a layer runs at on a plan whose fewest card experts a layer is `fewest`:
/// its target where a layer leaves it the spares and one that moves, else
/// half the fewest, the share the Qwen3.8 rule derives, never under
/// [`RESIDENCY_MIN_P`] nor past the ceiling; `None` where the card slots hold
/// neither. [`room_for`]'s card side, which [`residency_least_fewest`] reads
/// too.
fn card_p(target: usize, spares: usize, fewest: usize) -> Option<usize> {
    let ceiling = card_ceiling(fewest, spares);
    if target as isize <= ceiling {
        return Some(target);
    }
    let half = fewest / 2;
    (half >= RESIDENCY_MIN_P && half as isize <= ceiling).then_some(half)
}

/// The most pinned experts a layer of `fewest` card experts holds beside
/// `spares` free slots and one that moves; negative where it holds none.
fn card_ceiling(fewest: usize, spares: usize) -> isize {
    (fewest as isize)
        .saturating_sub(spares as isize)
        .saturating_sub(1)
}

/// The least fewest card experts a layer at which a word of `target` pinned
/// experts and `spares` free slots a layer runs on the card side of
/// [`room_for`]: below it a plan's room is [`RoomShort::NoRoom`], at it and
/// above the card slots hold the word. A context that leaves the card fewer
/// experts a layer than this runs no residency, so the context's own fit asks
/// it of its plan.
fn residency_least_fewest(target: usize, spares: usize) -> usize {
    let holds = target.saturating_add(spares).saturating_add(1);
    (1..=holds)
        .find(|&fewest| card_p(target, spares, fewest).is_some())
        .unwrap_or(holds)
}

/// The room a plan leaves a family's target residency word: `target` the
/// word's pinned experts, `spares` its free slots a layer, `card_experts`
/// the plan's card experts a layer (a layer holding none left out),
/// `pool_bytes` the churn pool's bytes at a pinned count (it shrinks as the
/// count grows), `headroom` the plan's host headroom and `mem_left` what the
/// host's room leaves past the plan's own host need. The word runs
/// at its target while its card slots hold it, its spares and one that
/// moves; where they do not, at
/// half the plan's fewest card experts a layer, never under
/// [`RESIDENCY_MIN_P`]. A load that takes its pool `unhosted`
/// ([`UnhostedPool::takes`]: the word asks for it with [`UNHOSTED_SUFFIX`]
/// and the body carries one) keeps P and leaves the pool unhosted
/// ([`Room::Unhosted`]), whether or not the pool fits. Where the churn pool
/// at P fits neither budget otherwise, the load runs at the first count
/// above P whose pool fits, never past what the card slots hold
/// ([`Room::MovedForPool`]), and `off` where none does
/// ([`RoomShort::PoolUnhostable`]): an unset word asks for nothing, so it
/// climbs on every body. A word that asks on a body that does not carry the
/// pool is the caller's refusal ([`UnhostedPool::refuse`]). Through [`Room::Moved`] and
/// [`Room::MovedForPool`] the word keeps its spares at the moved count;
/// through [`Room::Short`] it does not run — a default the user did not set
/// never refuses the load. An error of `pool_bytes` is the call's.
pub fn room_for<E>(
    unhosted: bool,
    target: usize,
    spares: usize,
    card_experts: impl IntoIterator<Item = u64>,
    pool_bytes: impl Fn(usize) -> Result<u64, E>,
    headroom: i128,
    mem_left: i128,
) -> Result<Room, E> {
    let Some(fewest) = card_experts.into_iter().filter(|&n| n > 0).min() else {
        return Ok(Room::Short(RoomShort::NoCardExperts));
    };
    let fewest = usize::try_from(fewest).unwrap_or(usize::MAX);
    // The card side: the word's P ([`card_p`]), at most the ceiling the
    // fewest layer leaves.
    let ceiling = card_ceiling(fewest, spares);
    let Some(p) = card_p(target, spares, fewest) else {
        return Ok(Room::Short(RoomShort::NoRoom { fewest }));
    };
    // The host side at P: the churn pool there fits the plan's host headroom
    // and what the host's room leaves past the plan's own host need.
    let fits = |needs: u64| headroom >= i128::from(needs) && mem_left >= i128::from(needs);
    let needs = pool_bytes(p)?;
    if unhosted {
        return Ok(Room::Unhosted {
            pinned: p,
            pool: PoolUnhosted {
                needs,
                headroom,
                mem_left,
            },
        });
    }
    if !fits(needs) {
        // The pool shrinks as the count grows, so the first count above the
        // word's whose pool fits is the nearest the sides leave; the probes
        // stop at the card side's ceiling.
        let mut needs = needs;
        for q in p + 1..=ceiling as usize {
            needs = pool_bytes(q)?;
            if fits(needs) {
                return Ok(Room::MovedForPool {
                    from: target,
                    to: q,
                });
            }
        }
        return Ok(Room::Short(RoomShort::PoolUnhostable(PoolUnhostable {
            needs,
            headroom,
            mem_left,
        })));
    }
    Ok(if p == target {
        Room::AsIs
    } else {
        Room::Moved {
            fewest,
            from: target,
            to: p,
        }
    })
}

/// The word `mid-p{p}-s{s}` as a pick's `&'static str`: leaked once a
/// resolution, as the set word's parse leaks it once a process
/// ([`Kind::Residency`]).
fn mid_word(p: usize, s: usize) -> &'static str {
    Box::leak(format!("mid-p{p}-s{s}").into_boxed_str())
}

/// [`RESIDENCY`] unset on a V4.1 plan: `pick` as it is unless it runs the
/// machine (`ResidencyWhy::Place` under a `mid-p<P>-s<S>` word); then
/// [`room_for`] the word's P and spares against the plan — through
/// [`ResidencyWhy::Shrunk`] the word runs at the P the card slots leave,
/// through [`ResidencyWhy::PoolShrunk`] at the first P whose pool the host
/// takes, `off` with why where the card slots or the host leave none, on
/// every body: an unset word never asks for its pool unhosted. `off` too
/// where the plan holds no host expert (`host_experts`, the routed experts it
/// leaves on no card; a plan with card experts only): the machine has none to
/// move onto the card and its spares would only displace card experts, the
/// clause [`residency38_at_plan`] holds for the Qwen3.8 word. A default the
/// user did not set never refuses the load. A set word and every other why
/// pass through untouched. An error of `pool_bytes` is the call's.
pub fn residency_at_plan<E>(
    carry: UnhostedPool,
    pick: ResidencyPick,
    card_experts: impl IntoIterator<Item = u64>,
    host_experts: u64,
    pool_bytes: impl Fn(usize) -> Result<u64, E>,
    headroom: i128,
    mem_left: i128,
) -> Result<ResidencyPick, E> {
    if pick.why != ResidencyWhy::Place {
        return Ok(pick);
    }
    let Some(ResidencyWord::Mid { pinned, spares, .. }) = residency_word(pick.word) else {
        return Ok(pick);
    };
    let layers: Vec<u64> = card_experts.into_iter().collect();
    if host_experts == 0 && layers.iter().any(|&n| n > 0) {
        return Ok(ResidencyPick {
            word: "off",
            why: ResidencyWhy::NoHostExperts,
        });
    }
    Ok(
        match room_for(
            carry.takes(false),
            pinned,
            spares,
            layers,
            pool_bytes,
            headroom,
            mem_left,
        )? {
            Room::AsIs => pick,
            Room::Moved { fewest, from, to } => ResidencyPick {
                word: mid_word(to, spares),
                why: ResidencyWhy::Shrunk { fewest, from, to },
            },
            Room::MovedForPool { from, to } => ResidencyPick {
                word: mid_word(to, spares),
                why: ResidencyWhy::PoolShrunk { from, to },
            },
            Room::Unhosted { pinned, pool } => ResidencyPick {
                word: mid_word(pinned, spares),
                why: ResidencyWhy::Unhosted(pool),
            },
            Room::Short(short) => ResidencyPick {
                word: "off",
                why: short.into(),
            },
        },
    )
}

/// The slots a layer the Qwen3.8 unset word frees for flips in flight.
pub const RESIDENCY38_SPARES: usize = 1;

/// The pinned experts a layer the Qwen3.8 unset word asks for before the
/// plan's room moves it ([`residency38_at_plan`]): none.
const RESIDENCY38_PINNED: usize = 0;

/// The fewest card experts a layer a plan holds for the Qwen3.8 unset word
/// to run the machine at all (the least fewest [`room_for`] holds its target
/// and spares at): the count a context's fit leaves the card its experts for,
/// since a plan under it resolves `off` ([`Residency38Why::NoRoom`]).
#[must_use]
pub fn residency38_least_fewest() -> u64 {
    let least = residency_least_fewest(RESIDENCY38_PINNED, RESIDENCY38_SPARES);
    u64::try_from(least).unwrap_or(u64::MAX)
}

/// The fewest card experts a layer a plan holds for a set `mid` word to run
/// as asked: its `pinned` experts, its `spares` and one that moves (the card
/// side of [`room_for`] at the word's own P, which a set word does not trade
/// for half the fewest); `None` for `off` and for a value the grammar does
/// not take ([`residency_word`]).
#[must_use]
pub fn residency_word_least_fewest(word: &str) -> Option<u64> {
    match residency_word(word)? {
        ResidencyWord::Mid { pinned, spares, .. } => {
            u64::try_from(pinned.saturating_add(spares).saturating_add(1)).ok()
        }
        ResidencyWord::Off => None,
    }
}

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
    /// Plan (a): no seed expert pinned; `fewest` the plan's fewest card
    /// experts a layer.
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
    /// The plan's card slots moved the word's pinned experts from `from` to
    /// `to`.
    Moved { from: usize, to: usize },
    /// The word asks for its pool unhosted ([`UNHOSTED_SUFFIX`]): it keeps
    /// its P and the load does not host the churn pool there
    /// ([`PoolUnhosted`]). No unset resolution gives it.
    Unhosted(PoolUnhosted),
    /// On a load that hosts its pool: the word's pinned experts raised from
    /// `from` to `to`, the first count whose churn pool fits the host.
    PoolShrunk { from: usize, to: usize },
    /// On a load that hosts its pool: no count the card slots hold has a
    /// churn pool that fits the host ([`PoolUnhostable`]).
    PoolUnhostable(PoolUnhostable),
}

impl fmt::Display for Residency38Why {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Residency38Why::PlanA { fewest } => write!(
                f,
                "unset: plan (a), no seed expert pinned (the plan's fewest card experts a layer \
                 is {fewest})"
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
            Residency38Why::Moved { from, to } => write!(
                f,
                "unset: the plan's room moves the word's pinned experts p{from}→p{to}"
            ),
            Residency38Why::Unhosted(pool) => pool.fmt(f),
            Residency38Why::PoolShrunk { from, to } => write!(
                f,
                "unset: p{from}→p{to} (host: the churn pool did not fit, raised)"
            ),
            Residency38Why::PoolUnhostable(pool) => pool.fmt(f),
        }
    }
}

impl From<RoomShort> for Residency38Why {
    fn from(short: RoomShort) -> Residency38Why {
        match short {
            RoomShort::NoCardExperts => Residency38Why::NoCardExperts,
            RoomShort::NoRoom { fewest } => Residency38Why::NoRoom { fewest },
            RoomShort::PoolUnhostable(pool) => Residency38Why::PoolUnhostable(pool),
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

    /// Whether the load hosts the word's churn pool: unhosted exactly where
    /// the plan's room left the pool out ([`Residency38Why::Unhosted`]), the
    /// one place a pick's hosting is derived.
    #[must_use]
    pub fn pool(&self) -> ChurnHosting {
        hosting(matches!(self.why, Residency38Why::Unhosted(_)))
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

/// Whether a Qwen3.8 load's [`RESIDENCY`] is `off` before its plan: `word` as
/// set, or unset where [`residency38_unset`] turns it `off` at `at`. Such a
/// load starts no expert stream, so it reserves no stream staging.
#[must_use]
pub fn residency38_off(word: Option<&str>, at: Residency38At) -> bool {
    word.map_or_else(|| residency38_unset(at).is_some(), |word| word == "off")
}

/// [`RESIDENCY`] unset on plan (a), from the plan: `card_experts` its card
/// experts a layer (a layer holding none left out), `host_experts` the
/// routed experts it leaves on no card, `pool_bytes` the churn pool's bytes
/// at P pinned, `headroom` the plan's host headroom and `mem_left` what the
/// host's room leaves past the plan's own host need and whatever the caller
/// counts beside it (the load's check before any upload; the Qwen3.8 room is
/// the placement crate's `HostNeed::room_left`: the room a plan is made at,
/// a read reading less the drift margin, past the plan's host need, which
/// already counts the checkpoints the load holds).
/// P is 0, no seed expert pinned — where the churn pool at 0 fits neither
/// budget, the first P whose pool fits ([`room_for`], on every body: an
/// unset word never asks for its pool unhosted); `off` when no layer holds a
/// card expert, when the host holds none (the pool would serve nothing),
/// when the fewest leave no room for P pinned, the spares and one that
/// moves, or when no P the card slots hold has a pool the host takes — a
/// default the user did not set never refuses the load. An error of
/// `pool_bytes` is the call's.
pub fn residency38_at_plan<E>(
    carry: UnhostedPool,
    card_experts: impl IntoIterator<Item = u64>,
    host_experts: u64,
    pool_bytes: impl Fn(usize) -> Result<u64, E>,
    headroom: i128,
    mem_left: i128,
) -> Result<Residency38Pick, E> {
    let layers: Vec<u64> = card_experts.into_iter().collect();
    let fewest = layers.iter().copied().filter(|&n| n > 0).min();
    let Some(fewest) = fewest else {
        return Ok(Residency38Pick::off(Residency38Why::NoCardExperts));
    };
    if host_experts == 0 {
        return Ok(Residency38Pick::off(Residency38Why::NoHostExperts));
    }
    let fewest = usize::try_from(fewest).unwrap_or(usize::MAX);
    Ok(
        match room_for(
            carry.takes(false),
            RESIDENCY38_PINNED,
            RESIDENCY38_SPARES,
            layers,
            pool_bytes,
            headroom,
            mem_left,
        )? {
            Room::AsIs => Residency38Pick {
                pinned: Some(0),
                why: Residency38Why::PlanA { fewest },
            },
            Room::Moved { from, to, .. } => Residency38Pick {
                pinned: Some(to),
                why: Residency38Why::Moved { from, to },
            },
            Room::MovedForPool { from, to } => Residency38Pick {
                pinned: Some(to),
                why: Residency38Why::PoolShrunk { from, to },
            },
            Room::Unhosted { pinned, pool } => Residency38Pick {
                pinned: Some(pinned),
                why: Residency38Why::Unhosted(pool),
            },
            Room::Short(short) => Residency38Pick::off(short.into()),
        },
    )
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
    /// The MTP source the run would open, and whether it is present.
    pub file: &'a Path,
    pub file_is_there: bool,
    /// Every path the unset pick checked, in order. Empty means the caller
    /// only has `file`, and the off line names that one path.
    pub looked: &'a [PathBuf],
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
    /// Unset, and no regular file at any path the pick checked. The paths are
    /// every place it looked, in order.
    NoFile(Vec<PathBuf>),
    /// Unset, and the last window needs `need` positions of `ctx`.
    Ctx { need: usize, ctx: usize },
    /// Unset on the Qwen3.8 seat, and the target's `name`, a matrix the
    /// draft borrows, is `ty` (or `absent`), not a form the draft reads.
    Borrowed { name: String, ty: String },
    /// Unset, and the draft yielded to the context
    /// ([`RungCost::of`]'s answer, the seats' plan-side rule).
    Yield(RungCost),
    /// Unset on the Qwen3.8 seat, and the plan pages host experts through
    /// the NVMe tier's RAM arena of `arena` bytes ([`paged_columns`]): the
    /// draft is off by default there (whether it pays depends on the
    /// machine's drive scaling with the queue depth a verify's fills run
    /// at); a set [`DRAFT`] runs.
    Paged { arena: u64 },
    /// Unset on the Qwen3.5 seat beside `--mmproj`: an image load gives every
    /// sequence a rotation table that draft's walks do not read, so it carries
    /// no draft; a set [`DRAFT`]`=mtp` is refused.
    Media,
    /// Unset on the Qwen3.8 seat serving a checkpoint folder without an MTP layer;
    /// a set [`DRAFT`]`=mtp` is refused.
    Folder,
    /// Unset on the Qwen3.8 seat serving a checkpoint folder whose MTP layer the draft
    /// does not read, `why` the layer's refusal; a set [`DRAFT`]`=mtp` is refused with it.
    FolderMtp { why: String },
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
            Draft38Off::NoFile(paths) => {
                let mut it = paths.iter();
                match it.next() {
                    None => f.write_str("no file"),
                    Some(first) => {
                        write!(f, "no file at {}", first.display())?;
                        for p in it {
                            write!(f, ", {}", p.display())?;
                        }
                        Ok(())
                    }
                }
            }
            Draft38Off::Ctx { need, ctx } => write!(
                f,
                "unset: the last window needs {need} positions, past --ctx {ctx}"
            ),
            Draft38Off::Borrowed { name, ty } => write!(
                f,
                "unset: the target's {name} is {ty}, not a form the MTP draft reads"
            ),
            Draft38Off::Yield(y) => y.fmt(f),
            Draft38Off::Paged { arena } => write!(
                f,
                "unset: the plan pages host experts through the NVMe tier's {arena} B RAM arena, \
                 and the draft is off by default there; set {DRAFT}=mtp to draft on a paged plan"
            ),
            Draft38Off::Media => f.write_str(
                "unset: --mmproj makes an image load, whose sequences hold rotation tables the \
                 MTP draft's walks do not read",
            ),
            Draft38Off::Folder => f.write_str("unset: the checkpoint folder holds no MTP layer"),
            Draft38Off::FolderMtp { why } => write!(
                f,
                "unset: the checkpoint folder's MTP layer is not one the draft reads: {why}"
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
        let paths = if at.looked.is_empty() {
            vec![at.file.to_path_buf()]
        } else {
            at.looked.to_vec()
        };
        Some(Draft38Off::NoFile(paths))
    } else if at.need > at.ctx {
        Some(Draft38Off::Ctx {
            need: at.need,
            ctx: at.ctx,
        })
    } else {
        None
    }
}

/// A reserve on the card that can cost an unset context positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reserve {
    /// An unset draft (Qwen3.8, GLM-5.3): it goes off when it costs a slot
    /// half the positions or more, and the `load draft=off` record's why is
    /// the cost.
    Draft,
    /// The image encoder (V4.1): it stays on, and the `ctx` line carries any
    /// cost.
    ImageEncoder,
}

/// What a reserve on the card cost the context: the positions a slot gets
/// with the reserve and without it, and the reserve's card bytes at `with`.
/// One owner of the comparison for every seat that weighs a reserve against
/// an unset context; [`Reserve`] says what the seat does with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RungCost {
    pub reserve: Reserve,
    /// The positions a slot gets with the reserve, 0 when that plan holds
    /// not even the context's floor.
    pub with: u64,
    /// The positions a slot gets without it.
    pub without: u64,
    /// The reserve's card bytes at `with`: the number the plan with it
    /// counted (Qwen3.8's `MtpInputs::card_bytes_of`, GLM-5.3's NextN card
    /// bytes with its arena, V4.1's encoder reserve).
    pub card_bytes: u64,
}

impl RungCost {
    /// `Some` when the reserve costs what its kind gives way to: the image
    /// encoder any position (`with` under `without`), the draft half a
    /// slot's positions or more (`with` at most half of `without`), so a
    /// draft that costs the fitted context a few steps stays on. The same
    /// context or a larger one costs nothing. A set `BLOOMERY_DRAFT=mtp`
    /// never asks: the seat runs it as given.
    #[must_use]
    pub fn of(reserve: Reserve, with: u64, without: u64, card_bytes: u64) -> Option<RungCost> {
        let costs = match reserve {
            Reserve::Draft => with.saturating_mul(2) <= without,
            Reserve::ImageEncoder => true,
        };
        (with < without && costs).then_some(RungCost {
            reserve,
            with,
            without,
            card_bytes,
        })
    }
}

impl fmt::Display for RungCost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (head, tail) = match self.reserve {
            Reserve::Draft => ("unset: the draft", "the draft"),
            Reserve::ImageEncoder => ("the image encoder", "the encoder"),
        };
        write!(
            f,
            "{head}'s {} B on the card leave a slot {} positions; without {tail} a slot takes {}",
            self.card_bytes, self.with, self.without
        )
    }
}

/// What the paged rule reads of a plan whose host experts page through the
/// NVMe tier's RAM arena ([`paged_columns`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PagedAt {
    /// The plan's RAM arena (`HostTotals::nvme_arena_bytes`); 0 is a plan
    /// the tier does not page through an arena.
    pub arena: u64,
    /// Whether the plan drafts, and whether [`DRAFT`] named the draft.
    pub draft: bool,
    pub draft_set: bool,
}

impl PagedAt {
    /// Whether the plan drafted before yielding to the context.
    #[must_use]
    pub fn asked_draft(&self, off: Option<&Draft38Off>) -> bool {
        self.draft || matches!(off, Some(Draft38Off::Yield(_)))
    }
}

/// Whether the paged rule turns the plan's draft off ([`paged_columns`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Paged {
    /// The plan as asked.
    AsAsked,
    /// The plan with the unset draft off ([`paged_draft`] names why; a set
    /// [`DRAFT`] runs).
    Reduced,
}

/// Whether paging turns an unset draft off. A plan with no arena, a plan
/// that does not draft, or a set draft, serves as asked; an unset draft on a
/// paged plan goes off. The slot count is not the rule's: a plan serves the
/// count it was asked. Whether a drafted
/// verify pays on a paged plan is the machine's drive, not the rule's, to
/// answer ([`DRAFT`]'s doc names the trade; `pageddraft-decode.card` holds
/// the measurement).
pub fn paged_columns(at: &PagedAt) -> Paged {
    if at.arena == 0 || !at.draft || at.draft_set {
        Paged::AsAsked
    } else {
        Paged::Reduced
    }
}

/// The draft a plan runs under the paged rule, and why a reduced load
/// ([`Paged::Reduced`]) runs without it, as its `load draft=off` record names
/// it: a plan with no arena ([`PagedAt::arena`] 0) keeps its draft and its
/// reason as asked; on a paged plan a set draft runs as asked; unset, a draft
/// that would run (`at.draft`), or that only its yield to the context turned
/// off — a verdict on the plan of several slots the load abandons — goes off
/// with [`Draft38Off::Paged`]; any other reason (set, the placement, the
/// file, a window past the context) holds whatever the plan, and stays. A
/// plan that would not draft is untouched. The Qwen3.8 seat's and
/// `generate_qwen3moe`'s one owner.
#[must_use]
pub fn paged_draft(at: &PagedAt, off: Option<Draft38Off>) -> (bool, Option<Draft38Off>) {
    if at.arena == 0 {
        return (at.draft, off);
    }
    let asked = at.asked_draft(off.as_ref());
    (
        at.draft && at.draft_set,
        if asked && !at.draft_set {
            Some(Draft38Off::Paged { arena: at.arena })
        } else {
            off
        },
    )
}

/// The word the GLM seat of `bloomery-serve` drafts by with [`DRAFT`]
/// unset where the draft can run: `mtp`, the file's NextN layer
/// ([`glm_unset`]).
pub const GLM_DRAFT_UNSET: &str = "mtp";

/// The word the GLM seat runs by with [`RESIDENCY`] unset where the machine
/// runs: the serving word ([`RESIDENCY_SERVING`]) ([`glm_unset`],
/// [`glm_residency_at_plan`]).
pub const GLM_RESIDENCY_UNSET: &str = RESIDENCY_SERVING;

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
    /// The plan's card slots moved the word's pinned experts from `from` to
    /// `to`.
    Moved { from: u64, to: u64 },
    /// The word asks for its pool unhosted ([`UNHOSTED_SUFFIX`]): it keeps
    /// its P and the load does not host the churn pool there
    /// ([`PoolUnhosted`]). No unset resolution gives it.
    Unhosted(PoolUnhosted),
    /// On a load that hosts its pool: the word's pinned experts raised from
    /// `from` to `to`, the first count whose churn pool fits the host.
    PoolShrunk { from: u64, to: u64 },
    /// On a load that hosts its pool: no count the card slots hold has a
    /// churn pool that fits the host ([`PoolUnhostable`]).
    PoolUnhostable(PoolUnhostable),
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
            GlmWhy::Moved { from, to } => write!(
                f,
                "unset: the plan's room moves the word's pinned experts p{from}→p{to}"
            ),
            GlmWhy::Unhosted(pool) => pool.fmt(f),
            GlmWhy::PoolShrunk { from, to } => write!(
                f,
                "unset: p{from}→p{to} (host: the churn pool did not fit, raised)"
            ),
            GlmWhy::PoolUnhostable(pool) => pool.fmt(f),
        }
    }
}

impl From<RoomShort> for GlmWhy {
    fn from(short: RoomShort) -> GlmWhy {
        match short {
            RoomShort::NoCardExperts => GlmWhy::NoCardExperts,
            RoomShort::NoRoom { fewest } => GlmWhy::NoRoom {
                fewest: fewest as u64,
            },
            RoomShort::PoolUnhostable(pool) => GlmWhy::PoolUnhostable(pool),
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

    /// Whether the load hosts the word's churn pool: unhosted exactly where
    /// the plan's room left the pool out ([`GlmWhy::Unhosted`]), the one
    /// place a pick's hosting is derived.
    #[must_use]
    pub fn pool(&self) -> ChurnHosting {
        hosting(matches!(self.why, GlmWhy::Unhosted(_)))
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
/// machine; then [`room_for`] the word's P and spares against the plan —
/// `off`, with why, when it holds no card expert (`card_experts` a layer, a
/// layer holding none left out) or when its fewest leave no room for the
/// word's pinned experts, its spares and one that moves; when the churn pool
/// there (`pool_bytes`) fits neither `headroom` (the plan's host headroom
/// less what the load hosts beside it) nor `mem_left` (what the host's room
/// leaves past the load's host need): the first P whose pool fits
/// ([`GlmWhy::PoolShrunk`]), or `off` where none does ([`room_for`], on
/// every body: an unset word never asks for its pool unhosted) — a default
/// the user did not set never refuses the load. An error of `pool_bytes` is
/// the call's.
pub fn glm_residency_at_plan<E>(
    carry: UnhostedPool,
    pick: GlmPick,
    card_experts: impl IntoIterator<Item = u64>,
    pool_bytes: impl Fn(usize) -> Result<u64, E>,
    headroom: i128,
    mem_left: i128,
) -> Result<GlmPick, E> {
    let Some(ResidencyWord::Mid { pinned, spares, .. }) = residency_word(pick.word) else {
        return Ok(pick);
    };
    Ok(
        match room_for(
            carry.takes(false),
            pinned,
            spares,
            card_experts,
            pool_bytes,
            headroom,
            mem_left,
        )? {
            Room::AsIs => pick,
            Room::Moved { from, to, .. } => GlmPick {
                word: mid_word(to, spares),
                why: GlmWhy::Moved {
                    from: from as u64,
                    to: to as u64,
                },
            },
            Room::MovedForPool { from, to } => GlmPick {
                word: mid_word(to, spares),
                why: GlmWhy::PoolShrunk {
                    from: from as u64,
                    to: to as u64,
                },
            },
            Room::Unhosted { pinned, pool } => GlmPick {
                word: mid_word(pinned, spares),
                why: GlmWhy::Unhosted(pool),
            },
            Room::Short(short) => GlmPick::off(short.into()),
        },
    )
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
    /// The load is made for a route trace ([`ROUTE_TRACE`] set,
    /// [`Levers::host`]): a load that can run a layer without its host leg
    /// (Qwen3.8, GLM-5.3) keeps every one, where the host service writes the
    /// trace, and holds a host view only of a layer with an expert off the
    /// card.
    pub route_trace: bool,
}

/// How the host tier's row dispatch lays its units over the pool
/// ([`HOST_LANES`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostLanes {
    /// Every matrix's units end to end in one space, one lane a participant.
    Flat,
    /// Each matrix's units laid over the pool's CCDs where the pool has a CCD
    /// map, flat where it has none.
    Ccd,
}

/// What an override cannot carry: a name read from the environment alone in
/// some file, which a value handed over would never reach.
const NO_OVERRIDE: &str = "a flag cannot set it: a reader of it takes its value from the \
                           environment only; set it in the environment";

/// What a second hand-over meets.
const OVERRIDE_ONCE: &str = "the process's overrides were handed over once already; a front \
                             hands them all in one call";

/// The Direct levers whose in-place reader asks [`override_of`] before the
/// environment, so an override reaches it. A Direct lever not listed is
/// refused as an override.
const READ_WITH_OVERRIDE: &[&str] = &["BLOOMERY_DSPARK_CARD"];

/// The overrides a front handed over ([`override_with`]), in the order given.
static OVERRIDES: OnceLock<Vec<(&'static str, String)>> = OnceLock::new();

/// `env` with `overrides` after it: a reading takes the last of a name given
/// twice, so an override replaces the environment's value.
fn merged(
    mut env: Vec<(OsString, OsString)>,
    overrides: &[(&'static str, String)],
) -> Vec<(OsString, OsString)> {
    env.extend(
        overrides
            .iter()
            .map(|(name, value)| (OsString::from(name), OsString::from(value))),
    );
    env
}

/// The environment a reading parses: the process's, with the overrides
/// ([`override_with`]) over it.
fn process_env() -> Vec<(OsString, OsString)> {
    merged(
        std::env::vars_os().collect(),
        OVERRIDES.get().map_or(&[][..], Vec::as_slice),
    )
}

/// The override handed over for `name`, for the in-place reader of a Direct
/// lever to take before it reads the environment; `None` where there is none.
/// Only the names of [`override_with`]'s Direct list have one.
#[must_use]
pub fn override_of(name: &str) -> Option<&'static str> {
    OVERRIDES
        .get()?
        .iter()
        .rev()
        .find(|(k, _)| *k == name)
        .map(|(_, v)| v.as_str())
}

/// The checked overrides, each row's name with its value, and the notes for
/// the environment values they replace.
type Overrides = (Vec<(&'static str, String)>, Vec<String>);

/// `pairs` as overrides of the environment's values, checked: each name a
/// row the override reaches, each value one its kind takes, every refusal at
/// once; and the notes for the environment values it replaces.
fn checked_overrides(
    pairs: &[(&str, String)],
    env: &[(OsString, OsString)],
) -> Result<Overrides, LeverError> {
    let mut set = Vec::new();
    let mut refused = Vec::new();
    for (name, value) in pairs {
        let refuse = |why: Why| Refusal {
            name: (*name).to_string(),
            value: value.clone(),
            why,
        };
        let Some(row) = spec(name) else {
            refused.push(refuse(Why::Unknown(nearest(name))));
            continue;
        };
        let taken = match row.site {
            Site::Retired { why, .. } => Err(Why::Retired(why)),
            Site::Parsed { left: [], .. } => row.kind.parse(value).map(drop).map_err(Why::Value),
            Site::Direct { .. } if READ_WITH_OVERRIDE.contains(&row.name) => {
                row.kind.taken(value).map(drop).map_err(Why::Value)
            }
            _ => Err(Why::Override(NO_OVERRIDE)),
        };
        match taken {
            Ok(()) => set.push((row.name, value.clone())),
            Err(why) => refused.push(refuse(why)),
        }
    }
    if !refused.is_empty() {
        return Err(LeverError { refused });
    }
    let notes = set
        .iter()
        .filter_map(|(name, value)| {
            let name: &str = name;
            let old = env.iter().rev().find(|(k, _)| k == name)?.1.as_os_str();
            (old != OsStr::new(value)).then(|| {
                format!(
                    "{name}: the environment's value {:?} is replaced by {value:?}",
                    lossy(old)
                )
            })
        })
        .collect();
    Ok((set, notes))
}

/// Values a front hands over in place of the environment's, once, before any
/// reading ([`at_main`] and the others, [`override_of`]): what a command-line
/// flag sets, which replaces what the environment holds. Each name must be a
/// row an override reaches: a Parsed lever no file reads in place, or a Direct
/// lever whose reader asks [`override_of`]; each value one its kind takes
/// (a Text lever's is its owner's to check). Refused, with every refusal at
/// once: a name no row names, a retired name, a name an override cannot
/// reach, a value its kind refuses, and a second call that hands over any.
/// The lines to print are the environment values replaced. The process
/// environment is read, never written.
pub fn override_with(pairs: &[(&str, String)]) -> Result<Vec<String>, LeverError> {
    let env: Vec<(OsString, OsString)> = std::env::vars_os().collect();
    let (set, notes) = checked_overrides(pairs, &env)?;
    if set.is_empty() {
        return Ok(notes);
    }
    OVERRIDES.set(set).map_err(|set| LeverError {
        refused: set
            .iter()
            .map(|(name, value)| Refusal {
                name: (*name).to_string(),
                value: value.clone(),
                why: Why::Override(OVERRIDE_ONCE),
            })
            .collect(),
    })?;
    Ok(notes)
}

/// A binary's one reading of the levers, first thing in `main`. `acts_on`
/// names the levers its run acts on: the Parsed ones (the pool's two,
/// [`THREADS`] and [`SPIN`], it always does) and the [`Readers::Named`]
/// Direct ones its path reads in place; a binary whose child inherits its
/// environment names the child's too. Refused, with every refusal at once: a
/// lever set to a value its kind does not take, read here or not; a Parsed
/// lever set when `acts_on` leaves it out; a retired name that is set; a
/// `BLOOMERY_*` name no row names. A Named Direct lever set when `acts_on`
/// leaves it out is no refusal, since one environment may carry the levers
/// of several models: it prints a line on stderr that it is not applied
/// ([`Reading::unapplied`]). A name in `acts_on` that is neither panics by
/// name: a caller's mistake. The environment it reads is the process's with
/// the values of [`override_with`] over it. With `--levers` among the
/// process's arguments it prints the reading's table on stdout (whether this
/// run applies each lever, and at what value) and exits 0, or returns the
/// reading's refusals after the table.
pub fn at_main(acts_on: &[&str]) -> Result<Levers, LeverError> {
    at_main_with_flags(acts_on, |name| {
        std::env::args_os().skip(1).any(|a| a == name)
    })
}

/// Read the levers as [`at_main`] does, querying only the caller's option
/// words for `--levers`, not values consumed by its argument parser.
pub fn at_main_with_flags(
    acts_on: &[&str],
    has_flag: impl FnOnce(&str) -> bool,
) -> Result<Levers, LeverError> {
    main_reading(acts_on, has_flag, false).0
}

/// Read the levers as [`at_main_with_flags`] does, for a run that prints the
/// lines of the levers it does not apply later, once: they are handed back
/// instead of printed, unless `--levers` prints them with its table.
pub fn at_main_held(
    acts_on: &[&str],
    has_flag: impl FnOnce(&str) -> bool,
) -> Result<(Levers, Vec<String>), LeverError> {
    let (levers, notes) = main_reading(acts_on, has_flag, true);
    Ok((levers?, notes))
}

fn main_reading(
    acts_on: &[&str],
    has_flag: impl FnOnce(&str) -> bool,
    held: bool,
) -> (Result<Levers, LeverError>, Vec<String>) {
    let bin = std::env::args_os()
        .next()
        .as_deref()
        .map(Path::new)
        .and_then(Path::file_name)
        .map_or_else(|| "unnamed".to_string(), lossy);
    let env = process_env();
    let reading = reading(&env, Scope::Main { bin: &bin, acts_on });
    let notes = reading.unapplied(&bin);
    let asked = has_flag("--levers");
    if !held || asked {
        for line in &notes {
            eprintln!("{line}");
        }
    }
    if asked {
        print!("{}", reading.table());
        if reading.refused.is_empty() {
            std::process::exit(0);
        }
    }
    (reading.levers(), notes)
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
    let env = process_env();
    let levers = read(&env, Scope::Pool)?;
    Ok(PoolLevers {
        threads: levers.threads(),
        spin: levers.spin(),
    })
}

/// The host's room ([`HOST_ROOM`]) and the NVMe expert tier's four levers
/// ([`NVTIER_BYTES`], [`NVTIER_READ`], [`NVTIER_MEM`], [`NVTIER_EVICT`]),
/// read where they act, below every binary's `main`: the host's available
/// bytes the room replaces, the placement dial the arena's bytes at the plan,
/// the tier's build the read mode, the arena's memory and its victim rule at
/// the load. [`PoolLevers`] is the shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NvTierLevers {
    /// `BLOOMERY_HOST_ROOM`: the host's room in bytes, given instead of
    /// read from the machine; `None` unset.
    pub host_room: Option<u64>,
    /// `BLOOMERY_NVTIER_BYTES`: the arena's bytes; `None` unset (the dial's
    /// default).
    pub bytes: Option<u64>,
    /// `BLOOMERY_NVTIER_READ`: whether the tier reads `O_DIRECT` (`direct`)
    /// or through the page cache (`buffered`).
    pub direct: bool,
    /// `BLOOMERY_NVTIER_MEM`: the memory the arena is.
    pub mem: NvTierMem,
    /// `BLOOMERY_NVTIER_EVICT`: which filled slot a miss takes.
    pub evict: NvTierEvict,
}

/// What `BLOOMERY_NVTIER_EVICT` asks a miss of the NVMe tier's arena to take
/// once no slot is free; both take the oldest transient first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NvTierEvict {
    /// `tick`, and unset: the least recently used hot slot.
    Tick,
    /// The hot slot whose expert has the least decayed routing count, the
    /// least recently used among equal counts.
    Heat,
}

/// What `BLOOMERY_NVTIER_MEM` asks the NVMe tier's arena to be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NvTierMem {
    /// `auto`, and unset: [`NvTierMem::Pinned`] on an integrated device,
    /// [`NvTierMem::Anon`] on a card with its own memory.
    Auto,
    /// An anonymous mapping, whose freed slots go back to the kernel.
    Anon,
    /// Pinned host memory the card maps.
    Pinned,
}

impl NvTierMem {
    /// Whether the arena is pinned on a device that is `integrated` (allocates
    /// from host memory) or not.
    #[must_use]
    pub fn pinned(self, integrated: bool) -> bool {
        match self {
            NvTierMem::Auto => integrated,
            NvTierMem::Anon => false,
            NvTierMem::Pinned => true,
        }
    }
}

/// The host room's and the NVMe expert tier's levers from the process
/// environment ([`NvTierLevers`]'s five), parsed as [`at_main`] parses them, and
/// refused, with every refusal at once, when one is set to a value its kind
/// does not take and when a retired name is set. Nothing else: which levers
/// a binary acts on, and which names it knows, is its `main`'s reading.
pub fn nvtier_levers() -> Result<NvTierLevers, LeverError> {
    let env = process_env();
    let levers = read(&env, Scope::NvTier)?;
    let bytes_of = |name: &str| match &levers.entry(name).value {
        None => None,
        Some(Value::Bytes(b)) => Some(*b),
        v => panic!("{name} holds {v:?}, not bytes"),
    };
    let (host_room, bytes) = (bytes_of(HOST_ROOM), bytes_of(NVTIER_BYTES));
    let direct = match &levers.entry(NVTIER_READ).value {
        None => true,
        Some(Value::Word("direct")) => true,
        Some(Value::Word("buffered")) => false,
        v => panic!("{NVTIER_READ} holds {v:?}, not direct or buffered"),
    };
    Ok(NvTierLevers {
        host_room,
        bytes,
        direct,
        mem: levers.nvtier_mem(),
        evict: levers.nvtier_evict(),
    })
}

/// What a load states of how it holds its host set, read where its plan
/// prints the need record and where the placed load consults the hook
/// ([`need_levers`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NeedLevers {
    /// [`NEED_HOOK`]: the executable a placed load calls before it opens a
    /// card; `None` unset, when nothing runs.
    pub hook: Option<PathBuf>,
    /// [`HOST_LOCK`]: the host set is mlocked for the model's life, which
    /// moves its pages from the cache class to the locked class in the
    /// need record.
    pub host_lock: bool,
}

/// [`NEED_HOOK`] and [`HOST_LOCK`] from the process environment, parsed as
/// [`at_main`] parses them, and refused, with every refusal at once, when one
/// is set to a value its kind does not take and when a retired name is set.
/// Nothing else: which levers a binary acts on, and which names it knows, is
/// its `main`'s reading, which refuses `HOST_LOCK` in a binary that does not
/// act on it, so the flag here is false there.
pub fn need_levers() -> Result<NeedLevers, LeverError> {
    let env = process_env();
    let levers = read(&env, Scope::Need)?;
    let hook = levers.path(NEED_HOOK).map(Path::to_path_buf);
    Ok(NeedLevers {
        hook,
        host_lock: levers.flag(HOST_LOCK),
    })
}

/// How the expert tier cards' page crosses ([`TIER_LINK`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TierLink {
    /// One page every card reaches.
    Shared,
    /// Two pages and a host pump thread between them.
    Loopback,
    /// Two pages, each with its own pump thread posting RDMA writes to the
    /// other.
    Rdma,
}

/// `BLOOMERY_TIER_LINK` from the process environment, parsed as [`at_main`]
/// parses it, and refused when it is set to a value its kind does not take
/// and when a retired name is set. Nothing else.
pub fn tier_link() -> Result<TierLink, LeverError> {
    let env = process_env();
    let levers = read(&env, Scope::TierLink)?;
    match &levers.entry(TIER_LINK).value {
        None | Some(Value::Word("shared")) => Ok(TierLink::Shared),
        Some(Value::Word("loopback")) => Ok(TierLink::Loopback),
        Some(Value::Word("rdma")) => Ok(TierLink::Rdma),
        v => panic!("{TIER_LINK} holds {v:?}, not shared, loopback or rdma"),
    }
}

/// Whether a tier's explicit listen or connect address is on a trusted local network:
/// loopback, private IPv4 (RFC 1918), or the benchmarking range 198.18.0.0/15
/// (RFC 2544), which fabric tools give back-to-back links between machines.
#[must_use]
pub fn tier_ip(ip: std::net::IpAddr) -> bool {
    ip.is_loopback()
        || matches!(ip, std::net::IpAddr::V4(v)
            if v.is_private() || (v.octets()[0] == 198 && v.octets()[1] & 0xfe == 18))
}

/// `BLOOMERY_TIER_NODES` from the process environment ([`TIER_NODES`]),
/// parsed as [`at_main`] parses it, and refused when it is set to a value its
/// kind does not take and when a retired name is set: the address of each
/// expert tier's node, in tier order; empty when unset. An off-host node requires RDMA.
pub fn tier_nodes() -> Result<Vec<SocketAddr>, LeverError> {
    let env = process_env();
    let levers = read(&env, Scope::TierNodes)?;
    match &levers.entry(TIER_NODES).value {
        None => Ok(Vec::new()),
        Some(Value::Addrs(a)) => Ok(a.clone()),
        v => panic!("{TIER_NODES} holds {v:?}, not socket addresses"),
    }
}

/// `BLOOMERY_CARD_BUDGET` from the process environment ([`CARD_BUDGET`]),
/// parsed as [`at_main`] parses it, and refused when it is set to a value its
/// kind does not take and when a retired name is set: read where a placement
/// is resolved against the census, which the ordinals of a per-card list are
/// checked against. Nothing else.
pub fn card_budget() -> Result<CardBudget, LeverError> {
    let env = process_env();
    Ok(read(&env, Scope::CardBudget)?.card_budget())
}

/// `BLOOMERY_CARD_MARGIN` from the process environment ([`CARD_MARGIN`]), read
/// as [`card_budget`] reads its lever, where a placement is resolved against
/// the census: the bytes each card keeps free, [`CardBudget::Unset`] unset.
pub fn card_margin() -> Result<CardBudget, LeverError> {
    let env = process_env();
    Ok(read(&env, Scope::CardBudget)?.card_margin())
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
