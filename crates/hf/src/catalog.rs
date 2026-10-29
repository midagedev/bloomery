//! The model catalog: the models bloomery starts by name, and for each one the files it can start
//! from. A row is a file of a Hugging Face repo at a pinned revision: a GGUF set (a quant tag of
//! the repo) or a safetensors checkpoint folder (the repo's one model, served as stored).
//!
//! The catalog is data and one rule. [`pick`] ranks a model's rows for a card class
//! ([`order`] of the placement's [`CardClass`]), leaves out what a start by name never takes on its own (a format that
//! is explicit-only, a row nobody has verified), and asks its caller for what only the machine can
//! say: whether the repo holds the row and how much is left to download ([`Avail`]), and whether
//! the seat's plan holds ([`Plan`]). The best-ranked row at the fastest placement tier wins: a plan
//! the cards hold whole, then one that needs the host's memory, then one paged from NVMe. Rows are
//! asked in rank order, and a row that cannot beat the best plan in hand is not asked: after a
//! plan the cards hold whole none can, after one that needs the host the rows too large for the
//! cards alone cannot. Such a row costs no listing, no header and no plan.
//!
//! `tools/catalog.tsv` is [`print`]'s committed copy.

mod rows;

use std::cmp::Reverse;
use std::fmt;
use std::path::Path;

use bloomery_placement::placement::devices::CardClass;

use crate::{Entry, HfError};

/// The revision of a row whose repo's tables pin no commit.
pub const FLOATING: &str = "main";

/// How a row's files are stored in its repo.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A GGUF set of the repo; the row's variant is its quant tag (`--hf <repo>:<variant>`).
    Gguf,
    /// A safetensors checkpoint folder, the repo's one model (`--hf <repo>`).
    Folder,
}

/// What a row's weights are stored as: the key the card classes order by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// NVIDIA's FP4-Local folder: NVFP4 experts with compressed block scales.
    Fp4Local,
    /// A ModelOpt NVFP4 folder.
    Nvfp4,
    /// A GGUF set of MXFP4 weights.
    Mxfp4,
    /// A GGUF set of 8-bit weights (`Q8_0`).
    Gguf8,
    /// A GGUF set of 6-bit weights (`Q6_K`).
    Gguf6,
    /// A GGUF set of 5-bit weights (`Q5_K_M`).
    Gguf5,
    /// A GGUF set of 4-bit weights (`Q4_K_M`, `UD-Q4_K_XL`).
    Gguf4,
    /// A GGUF set of 3-bit weights (`Q3_K_M`, `UD-Q3_K_XL`).
    Gguf3,
    /// A GGUF set of 2-bit weights (`UD-IQ2_XXS`).
    Gguf2,
}

impl Format {
    /// Every format, the explicit-only ones included.
    pub const ALL: [Format; 9] = [
        Format::Fp4Local,
        Format::Nvfp4,
        Format::Mxfp4,
        Format::Gguf8,
        Format::Gguf6,
        Format::Gguf5,
        Format::Gguf4,
        Format::Gguf3,
        Format::Gguf2,
    ];

    /// Whether a start by name takes a row of this format on its own. The 8-, 6- and 5-bit sets
    /// give speed up for accuracy, so only a start that names the row takes them.
    pub const fn automatic(self) -> bool {
        !matches!(self, Format::Gguf8 | Format::Gguf6 | Format::Gguf5)
    }

    /// The spelling `tools/catalog.tsv` and the JSON contract use.
    pub const fn name(self) -> &'static str {
        match self {
            Format::Fp4Local => "fp4-local",
            Format::Nvfp4 => "nvfp4",
            Format::Mxfp4 => "mxfp4",
            Format::Gguf8 => "gguf8",
            Format::Gguf6 => "gguf6",
            Format::Gguf5 => "gguf5",
            Format::Gguf4 => "gguf4",
            Format::Gguf3 => "gguf3",
            Format::Gguf2 => "gguf2",
        }
    }
}

/// Where a native FP4 product runs, the FP4 folders come first, the FP4-Local folder before the
/// NVFP4 one (the faster of the two on the Spark's rows).
const FP4_FIRST: &[Format] = &[
    Format::Fp4Local,
    Format::Nvfp4,
    Format::Gguf4,
    Format::Mxfp4,
    Format::Gguf3,
    Format::Gguf2,
];

/// Without native FP4 products an FP4 folder gains nothing from its format, so it ranks after
/// every GGUF.
const GGUF_FIRST: &[Format] = &[
    Format::Gguf4,
    Format::Mxfp4,
    Format::Gguf3,
    Format::Gguf2,
    Format::Fp4Local,
    Format::Nvfp4,
];

/// The automatic formats of `class`, best first; a row whose format is not here is never ranked.
pub const fn order(class: CardClass) -> &'static [Format] {
    match class {
        CardClass::Ampere => GGUF_FIRST,
        CardClass::Blackwell | CardClass::Gb10 => FP4_FIRST,
    }
}

/// The spelling the JSON contract uses for `class`.
pub const fn class_name(class: CardClass) -> &'static str {
    match class {
        CardClass::Ampere => "ampere",
        CardClass::Blackwell => "blackwell",
        CardClass::Gb10 => "gb10",
    }
}

/// Whether the maintainers run a row's file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// The file is one the maintainers run; `evidence` says where the repo records it.
    Verified { evidence: &'static str },
    /// Listed and startable by name, never picked on its own; `why` says what is missing.
    Unverified { why: &'static str },
}

impl Status {
    /// Whether the row is verified.
    pub const fn is_verified(self) -> bool {
        matches!(self, Status::Verified { .. })
    }

    /// The evidence of a verified row, or what an unverified one lacks.
    pub const fn note(self) -> &'static str {
        match self {
            Status::Verified { evidence } => evidence,
            Status::Unverified { why } => why,
        }
    }
}

/// One file of a repo a model can start from.
#[derive(Debug, PartialEq, Eq)]
pub struct Row {
    /// The quant tag of a GGUF set, or the folder's short name (`nvfp4`, `fp4-local`).
    pub variant: &'static str,
    pub kind: Kind,
    /// The Hugging Face repo, `<owner>/<name>`.
    pub repo: &'static str,
    /// A commit of the repo, or [`FLOATING`] where the repo's tables pin none.
    pub revision: &'static str,
    pub format: Format,
    /// Decimal bytes of the files: exact where a table of the repo holds them, else the docs'
    /// figure, `None` where neither does. An estimate the listing replaces.
    pub bytes: Option<u64>,
    pub status: Status,
}

impl Row {
    /// The `--hf` value that names this row's files.
    pub fn hf(&self) -> String {
        match self.kind {
            Kind::Gguf => format!("{}:{}", self.repo, self.variant),
            Kind::Folder => self.repo.to_owned(),
        }
    }

    /// Whether the row's revision is a commit rather than [`FLOATING`].
    pub fn is_pinned(&self) -> bool {
        self.revision != FLOATING
    }
}

/// A model the catalog names and the rows it can start from.
#[derive(Debug, PartialEq, Eq)]
pub struct Model {
    /// The catalog key, lowercase.
    pub name: &'static str,
    /// Other spellings; a seat word is one for the single model that is its seat's default.
    pub aliases: &'static [&'static str],
    /// The `--model` word of the seat that serves the model.
    pub seat: &'static str,
    /// A `general.architecture` string of `tools/arch-registry.tsv`.
    pub arch: &'static str,
    pub rows: &'static [Row],
}

impl Model {
    /// The row named `variant`, ignoring ASCII case.
    pub fn row(&self, variant: &str) -> Option<&'static Row> {
        let rows: &'static [Row] = self.rows;
        rows.iter()
            .find(|row| row.variant.eq_ignore_ascii_case(variant))
    }

    fn answers_to(&self, name: &str) -> bool {
        self.name.eq_ignore_ascii_case(name)
            || self
                .aliases
                .iter()
                .any(|alias| alias.eq_ignore_ascii_case(name))
    }
}

/// Why a catalog target was refused.
#[derive(Debug, thiserror::Error)]
pub enum CatalogError {
    #[error("model {name:?} is not in the catalog; its models: {}", .names.join(", "))]
    UnknownModel {
        name: String,
        names: Vec<&'static str>,
    },
    #[error("{model} has no variant {variant:?}; its variants: {}", .variants.join(", "))]
    UnknownVariant {
        model: &'static str,
        variant: String,
        variants: Vec<&'static str>,
    },
    #[error("{0:?}: a catalog target is NAME or NAME:VARIANT, each part non-empty")]
    Empty(String),
}

/// Every model of the catalog.
pub fn models() -> &'static [Model] {
    rows::MODELS
}

/// The model `name` is, by its name, an alias or its seat word, ignoring ASCII case.
pub fn find(name: &str) -> Result<&'static Model, CatalogError> {
    models()
        .iter()
        .find(|model| model.answers_to(name))
        .ok_or_else(|| CatalogError::UnknownModel {
            name: name.to_owned(),
            names: models().iter().map(|model| model.name).collect(),
        })
}

/// A catalog model and, when the target names one, its variant: `NAME` or `NAME:VARIANT`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CatalogRef {
    pub model: &'static Model,
    /// The row's own spelling of the variant the target named.
    pub variant: Option<&'static str>,
}

impl CatalogRef {
    /// `NAME[:VARIANT]` where NAME is a model's name, alias or seat word and VARIANT one of its
    /// rows, each ignoring ASCII case. An unknown name or variant is refused listing the known
    /// ones.
    pub fn parse(s: &str) -> Result<CatalogRef, CatalogError> {
        let (name, variant) = match s.split_once(':') {
            Some((name, variant)) => (name, Some(variant)),
            None => (s, None),
        };
        if name.is_empty() || variant == Some("") {
            return Err(CatalogError::Empty(s.to_owned()));
        }
        let model = find(name)?;
        let variant = variant
            .map(|v| {
                model
                    .row(v)
                    .map(|row| row.variant)
                    .ok_or_else(|| CatalogError::UnknownVariant {
                        model: model.name,
                        variant: v.to_owned(),
                        variants: model.rows.iter().map(|row| row.variant).collect(),
                    })
            })
            .transpose()?;
        Ok(CatalogRef { model, variant })
    }
}

impl fmt::Display for CatalogRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.variant {
            Some(variant) => write!(f, "{}:{variant}", self.model.name),
            None => f.write_str(self.model.name),
        }
    }
}

/// What a pick reads of the machine: the card class that orders the formats, the memory of its
/// cards and the room the download cache has.
#[derive(Clone, Copy, Debug)]
pub struct Machine {
    pub class: CardClass,
    /// The memory of every card together: a row whose files are larger than this cannot be held
    /// by the cards alone ([`Tier::Cards`]).
    pub card_bytes: u64,
    pub cache_free_bytes: u64,
}

/// What the repo's listing and the cache say of one row's files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Avail {
    /// The repo holds the files, `download_bytes` of them still to fetch: none when the cache
    /// holds a verified set.
    Present { download_bytes: u64 },
    /// The listing lacks the row's set.
    NotInRepo,
    /// No listing could be read, and no verified set is cached.
    OfflineUncached,
}

/// Where a plan puts the weights, fastest first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// The cards hold every weight.
    Cards,
    /// The host's memory holds the rest.
    Host,
    /// Part of the rest is paged from NVMe.
    Paged,
}

impl Tier {
    /// The spelling a rejection word and the JSON contract use.
    pub const fn name(self) -> &'static str {
        match self {
            Tier::Cards => "cards",
            Tier::Host => "host",
            Tier::Paged => "paged",
        }
    }
}

/// How sure a plan is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fit {
    /// The seat planned the row's own header.
    Planned,
    /// Decided from bytes alone, no header read.
    Estimate,
}

/// The seat's plan for one row, when it holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub place: String,
    pub ctx: u64,
    pub slots: u64,
    /// Weight bytes the plan leaves on the host.
    pub host_bytes: u64,
    /// Of those, the bytes paged from NVMe.
    pub paged_bytes: u64,
    pub fit: Fit,
}

impl Plan {
    /// The slowest place the plan puts weights.
    pub fn tier(&self) -> Tier {
        if self.paged_bytes > 0 {
            Tier::Paged
        } else if self.host_bytes > 0 {
            Tier::Host
        } else {
            Tier::Cards
        }
    }
}

/// The seat's refusal of one row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    /// The refusal's code, as the plan's JSON names it.
    pub code: String,
    /// Bytes the plan asked for, where it says.
    pub need_bytes: Option<u64>,
}

/// Why a row was not picked: one word each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reason {
    /// The row's format is explicit-only and the start did not name the row.
    ExplicitOnly,
    /// Nobody has verified the row and the start did not name it.
    Unverified,
    /// The repo's listing lacks the row's set.
    NotInRepo,
    /// No listing could be read and no verified set is cached.
    OfflineUncached,
    /// The download is larger than the cache's free room.
    DiskShort,
    /// The seat's plan refused the row.
    Plan {
        code: String,
        need_bytes: Option<u64>,
    },
    /// The row holds, but only at this slower tier, while a lower-ranked row holds at a faster one.
    Slower(Tier),
    /// A better-ranked row was picked, or ranks above it at the same or a faster tier. The row
    /// below a pick is not asked about: it may or may not hold.
    RankedBelow,
}

impl fmt::Display for Reason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Reason::ExplicitOnly => f.write_str("explicit-only"),
            Reason::Unverified => f.write_str("unverified"),
            Reason::NotInRepo => f.write_str("not-in-repo"),
            Reason::OfflineUncached => f.write_str("offline-uncached"),
            Reason::DiskShort => f.write_str("disk-short"),
            Reason::Plan { code, .. } => write!(f, "plan:{code}"),
            Reason::Slower(tier) => f.write_str(tier.name()),
            Reason::RankedBelow => f.write_str("ranked-below"),
        }
    }
}

/// The row a pick chose.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Picked {
    pub row: &'static Row,
    pub plan: Plan,
    /// Bytes still to fetch; 0 when the cache holds a verified set.
    pub download_bytes: u64,
}

/// A row a pick did not choose and the one word why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rejected {
    pub row: &'static Row,
    pub why: Reason,
}

/// What a pick decided: the chosen row, if any, and every other candidate with its reason, in the
/// model's row order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome {
    pub picked: Option<Picked>,
    pub rejected: Vec<Rejected>,
}

impl Outcome {
    /// The least `need_bytes` among the seat's refusals, the figure a no-fit refusal names.
    pub fn least_need_bytes(&self) -> Option<u64> {
        self.rejected
            .iter()
            .filter_map(|rejected| match rejected.why {
                Reason::Plan { need_bytes, .. } => need_bytes,
                _ => None,
            })
            .min()
    }
}

/// The best-ranked row of `target`'s model that holds on `machine`, at the fastest tier that has
/// one, and every other candidate with a reason. The candidates are the named variant alone, or
/// every row of the model; a start that names no variant leaves out the explicit-only formats and
/// the unverified rows. The rows are asked in rank order: `avail` says what the listing and the
/// cache hold of a row, `probe` plans it. A row that cannot beat the best plan in hand is not
/// asked ([`ceiling`]) and is [`Reason::RankedBelow`].
pub fn pick(
    target: &CatalogRef,
    machine: &Machine,
    avail: &mut dyn FnMut(&'static Row) -> Avail,
    probe: &mut dyn FnMut(&'static Row) -> Result<Plan, Refusal>,
) -> Outcome {
    let model = target.model;
    let named = target.variant.is_some();
    let mut order: Vec<&'static Row> = model
        .rows
        .iter()
        .filter(|row| target.variant.is_none_or(|variant| row.variant == variant))
        .collect();
    order.sort_by_key(|row| (rank(machine.class, row), Reverse(row.bytes.unwrap_or(0))));

    let mut rejected: Vec<Rejected> = Vec::new();
    let mut held: Vec<(&'static Row, u64, Plan)> = Vec::new();
    let mut best: Option<Tier> = None;
    for row in order {
        if let Err(why) = policy(row, named) {
            rejected.push(Rejected { row, why });
            continue;
        }
        if best.is_some_and(|best| ceiling(row, machine) >= best) {
            rejected.push(Rejected {
                row,
                why: Reason::RankedBelow,
            });
            continue;
        }
        let download_bytes = match screen(row, machine, avail) {
            Ok(bytes) => bytes,
            Err(why) => {
                rejected.push(Rejected { row, why });
                continue;
            }
        };
        match probe(row) {
            Ok(plan) => {
                let tier = plan.tier();
                best = Some(best.map_or(tier, |best| best.min(tier)));
                held.push((row, download_bytes, plan));
            }
            Err(refusal) => rejected.push(Rejected {
                row,
                why: Reason::Plan {
                    code: refusal.code,
                    need_bytes: refusal.need_bytes,
                },
            }),
        }
    }

    // `held` is in rank order: the first row at the fastest tier wins, the held rows above it are
    // slower, and the ones below it rank below.
    let mut picked: Option<Picked> = None;
    for (row, download_bytes, plan) in held {
        let tier = plan.tier();
        if picked.is_some() {
            rejected.push(Rejected {
                row,
                why: Reason::RankedBelow,
            });
        } else if best == Some(tier) {
            picked = Some(Picked {
                row,
                plan,
                download_bytes,
            });
        } else {
            rejected.push(Rejected {
                row,
                why: Reason::Slower(tier),
            });
        }
    }
    rejected.sort_by_key(|r| model.rows.iter().position(|m| m.variant == r.row.variant));
    Outcome { picked, rejected }
}

/// The fastest tier `row` can reach on `machine` without asking anything: the cards unless its
/// files are larger than all the cards' memory together, then the host's. A row of unknown size
/// may reach the cards.
fn ceiling(row: &Row, machine: &Machine) -> Tier {
    if row.bytes.is_some_and(|bytes| bytes > machine.card_bytes) {
        Tier::Host
    } else {
        Tier::Cards
    }
}

/// The row's place in the class's order; a format the order does not hold ranks last.
fn rank(class: CardClass, row: &Row) -> usize {
    order(class)
        .iter()
        .position(|format| *format == row.format)
        .unwrap_or(usize::MAX)
}

/// Whether a start may take `row` on its own: an explicit-only format and an unverified row only
/// when the start names it.
fn policy(row: &Row, named: bool) -> Result<(), Reason> {
    if named {
        return Ok(());
    }
    if !row.format.automatic() {
        return Err(Reason::ExplicitOnly);
    }
    if !row.status.is_verified() {
        return Err(Reason::Unverified);
    }
    Ok(())
}

/// The bytes still to fetch of a row that may be probed, or the reason it may not.
fn screen(
    row: &'static Row,
    machine: &Machine,
    avail: &mut dyn FnMut(&'static Row) -> Avail,
) -> Result<u64, Reason> {
    match avail(row) {
        Avail::NotInRepo => Err(Reason::NotInRepo),
        Avail::OfflineUncached => Err(Reason::OfflineUncached),
        Avail::Present { download_bytes } if download_bytes > machine.cache_free_bytes => {
            Err(Reason::DiskShort)
        }
        Avail::Present { download_bytes } => Ok(download_bytes),
    }
}

impl Fit {
    /// The spelling the JSON contract uses.
    pub const fn name(self) -> &'static str {
        match self {
            Fit::Planned => "planned",
            Fit::Estimate => "estimate",
        }
    }
}

/// The files of `row` in the repo's listing `entries`: the GGUF set its quant tag names, or the
/// checkpoint its index holds. `None` when the listing lacks the set; a repo the listing makes
/// ambiguous or incomplete is refused by name ([`HfError`]).
pub fn row_files(row: &Row, entries: &[Entry]) -> Result<Option<Vec<Entry>>, HfError> {
    match row.kind {
        Kind::Gguf => {
            let sets = crate::gguf_sets(row.repo, entries)?;
            match crate::pick(row.repo, &sets, Some(row.variant)) {
                Ok(set) => Ok(Some(set.files.clone())),
                Err(HfError::NoGguf { .. } | HfError::NoMatch { .. }) => Ok(None),
                Err(e) => Err(e),
            }
        }
        Kind::Folder => Ok(crate::checkpoint_files(entries)),
    }
}

/// Whether `cached` (the verified files of the repo's cache) holds `file`: a verified file with
/// its path, size and digest.
#[must_use]
pub fn is_cached(file: &Entry, cached: &[Entry]) -> bool {
    cached
        .iter()
        .any(|c| c.path == file.path && c.size == file.size && c.digest == file.digest)
}

/// What the repo's listing `entries` and the cache's verified files `cached` say of `row`'s files:
/// the bytes of [`row_files`] the cache does not hold ([`is_cached`]), or [`Avail::NotInRepo`]
/// when the listing lacks the set.
pub fn availability(row: &Row, entries: &[Entry], cached: &[Entry]) -> Result<Avail, HfError> {
    let Some(files) = row_files(row, entries)? else {
        return Ok(Avail::NotInRepo);
    };
    let download_bytes = files
        .iter()
        .filter(|file| !is_cached(file, cached))
        .map(|file| file.size)
        .sum();
    Ok(Avail::Present { download_bytes })
}

/// [`availability`] when no listing could be read: the verified `cached` files are all there is,
/// so a row whose set they hold whole is present with nothing to fetch and every other row is
/// [`Avail::OfflineUncached`].
pub fn availability_offline(row: &Row, cached: &[Entry]) -> Avail {
    match availability(row, cached, cached) {
        Ok(avail @ Avail::Present { .. }) => avail,
        Ok(Avail::NotInRepo | Avail::OfflineUncached) | Err(_) => Avail::OfflineUncached,
    }
}

/// The bytes free to this process on the filesystem that holds the cache `root` (read at its
/// nearest existing ancestor on a first start): the room a download may take.
pub fn cache_free_bytes(root: &Path) -> Result<u64, HfError> {
    let at = root.ancestors().find(|a| a.exists()).unwrap_or(root);
    os::free_bytes(at).map_err(|e| match e {
        os::OsError::Io(err) => HfError::Io {
            path: at.display().to_string(),
            err,
        },
        os::OsError::Unsupported { call } => HfError::Source(format!(
            "{call}: this platform cannot read the cache's free bytes"
        )),
    })
}

/// The variant of `model` one step smaller than `row` on `class`: among the rows an automatic
/// start may take (an automatic format, verified, a known size), the largest one under `row`'s
/// bytes. `None` when `row`'s size is unknown or no row is smaller.
pub fn next_smaller(model: &'static Model, class: CardClass, row: &Row) -> Option<&'static Row> {
    let have = row.bytes?;
    let rows: &'static [Row] = model.rows;
    rows.iter()
        .filter(|r| r.format.automatic() && r.status.is_verified())
        .filter(|r| order(class).contains(&r.format))
        .filter(|r| r.bytes.is_some_and(|bytes| bytes < have))
        .max_by_key(|r| r.bytes)
}

/// The schema a `--plan --json` line names.
pub const PLAN_SCHEMA: &str = "bloomery.plan/1";

/// The seat's answer to a `--plan --json` run, read from its `text` (stdout: the plan's records
/// may share it, so the last line that is a JSON object naming [`PLAN_SCHEMA`] is the answer):
/// the plan, or its refusal. No such line, or one whose fields are not the contract's, is an
/// error naming what was missing.
pub fn read_plan(text: &str) -> Result<Result<Plan, Refusal>, String> {
    let line = text
        .lines()
        .rev()
        .filter(|l| l.starts_with('{'))
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .find(|v| v["schema"] == PLAN_SCHEMA)
        .ok_or_else(|| format!("no line of the output names the schema {PLAN_SCHEMA}"))?;
    if let Some(refused) = line.get("refused") {
        let code = refused["code"]
            .as_str()
            .ok_or("the refusal names no code")?
            .to_owned();
        return Ok(Err(Refusal {
            code,
            need_bytes: refused["need"].as_u64(),
        }));
    }
    let number = |key: &str| {
        line[key]
            .as_u64()
            .ok_or_else(|| format!("the plan's {key} is not a number"))
    };
    Ok(Ok(Plan {
        place: line["place"]
            .as_str()
            .ok_or("the plan names no place")?
            .to_owned(),
        ctx: number("ctx")?,
        slots: number("slots")?,
        host_bytes: number("host_bytes")?,
        paged_bytes: number("paged")?,
        fit: Fit::Planned,
    }))
}

/// The chosen row as the keys of the JSON contract's `pick` object.
pub fn pick_fields(picked: &Picked) -> serde_json::Map<String, serde_json::Value> {
    let mut fields = serde_json::Map::new();
    let values = [
        ("variant", serde_json::json!(picked.row.variant)),
        ("source", serde_json::json!(picked.row.hf())),
        ("revision", serde_json::json!(picked.row.revision)),
        ("download", serde_json::json!(picked.download_bytes)),
        ("cached", serde_json::json!(picked.download_bytes == 0)),
        ("place", serde_json::json!(picked.plan.place)),
        ("ctx", serde_json::json!(picked.plan.ctx)),
        ("slots", serde_json::json!(picked.plan.slots)),
        ("paged", serde_json::json!(picked.plan.paged_bytes)),
        ("fit", serde_json::json!(picked.plan.fit.name())),
    ];
    for (key, value) in values {
        fields.insert(key.to_owned(), value);
    }
    fields
}

/// The chosen row as the JSON contract's `pick` object.
pub fn pick_json(picked: &Picked) -> serde_json::Value {
    serde_json::Value::Object(pick_fields(picked))
}

/// One model's entry of the JSON contract: its name and seat, the pick (`null` when nothing
/// holds) and every other candidate with its one reason word.
pub fn model_json(model: &Model, outcome: &Outcome) -> serde_json::Value {
    serde_json::json!({
        "name": model.name,
        "seat": model.seat,
        "pick": outcome.picked.as_ref().map(pick_json),
        "rejected": outcome
            .rejected
            .iter()
            .map(|r| serde_json::json!({"variant": r.row.variant, "why": r.why.to_string()}))
            .collect::<Vec<_>>(),
    })
}

/// The catalog as `tools/catalog.tsv` holds it: a comment line, a header line, then one
/// tab-separated line per row: the model's name, its aliases (comma-separated, `-` for none), seat
/// and architecture, then the row's variant, format, `--hf` value, revision, bytes (`-` for none),
/// whether it is verified, and the evidence or what is missing.
pub fn print() -> String {
    let mut out = String::from(
        "# the rows of hf::catalog, as `hf::catalog::print` prints them\n\
         name\taliases\tseat\tarch\tvariant\tformat\thf\trevision\tbytes\tverified\tnote\n",
    );
    for model in models() {
        let aliases = if model.aliases.is_empty() {
            "-".to_owned()
        } else {
            model.aliases.join(",")
        };
        for row in model.rows {
            let bytes = row
                .bytes
                .map_or_else(|| "-".to_owned(), |bytes| bytes.to_string());
            out.push_str(&format!(
                "{}\t{aliases}\t{}\t{}\t{}\t{}\t{}\t{}\t{bytes}\t{}\t{}\n",
                model.name,
                model.seat,
                model.arch,
                row.variant,
                row.format.name(),
                row.hf(),
                row.revision,
                row.status.is_verified(),
                row.status.note(),
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARCH_REGISTRY: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/arch-registry.tsv");
    const CATALOG: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/catalog.tsv");
    const CENSUS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/census/stubs.tsv");
    const SERVE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../gpu-gates/src/bin/bloomery_serve.rs"
    );
    /// Room for any download.
    const ROOM: u64 = 1_000_000_000_000;

    fn read(path: &str) -> String {
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"))
    }

    /// A plan that holds at `tier`.
    fn plan(tier: Tier) -> Result<Plan, Refusal> {
        let (host_bytes, paged_bytes) = match tier {
            Tier::Cards => (0, 0),
            Tier::Host => (1, 0),
            Tier::Paged => (1, 1),
        };
        Ok(Plan {
            place: "a".to_owned(),
            ctx: 4096,
            slots: 1,
            host_bytes,
            paged_bytes,
            fit: Fit::Planned,
        })
    }

    /// A plan the seat refuses with `code`, asking `need_bytes`.
    fn refused(code: &str, need_bytes: u64) -> Result<Plan, Refusal> {
        Err(Refusal {
            code: code.to_owned(),
            need_bytes: Some(need_bytes),
        })
    }

    /// One pick of `target` on a machine of `class` with `cache_free_bytes` of cache room, the
    /// listing as `avail` says (a row it does not name is present and cached) and the probe as
    /// `answers` say (a row it does not name is a failure). The outcome, and the variants the
    /// probe was asked about, in the order it was asked.
    fn run(
        target: &str,
        class: CardClass,
        cache_free_bytes: u64,
        avail: &[(&str, Avail)],
        answers: &[(&str, Result<Plan, Refusal>)],
    ) -> (Outcome, Vec<&'static str>) {
        let target = CatalogRef::parse(target).unwrap_or_else(|e| panic!("{target}: {e}"));
        let machine = Machine {
            class,
            card_bytes: ROOM,
            cache_free_bytes,
        };
        let mut probed: Vec<&'static str> = Vec::new();
        let outcome = pick(
            &target,
            &machine,
            &mut |row| {
                avail
                    .iter()
                    .find(|(variant, _)| *variant == row.variant)
                    .map_or(Avail::Present { download_bytes: 0 }, |&(_, a)| a)
            },
            &mut |row| {
                probed.push(row.variant);
                answers
                    .iter()
                    .find(|(variant, _)| *variant == row.variant)
                    .map(|(_, answer)| answer.clone())
                    .unwrap_or_else(|| panic!("no probe answer for {}", row.variant))
            },
        );
        // Every candidate is picked or rejected, once, with its word.
        let candidates = target.variant.map_or(target.model.rows.len(), |_| 1);
        let mut seen: Vec<&str> = outcome.rejected.iter().map(|r| r.row.variant).collect();
        seen.extend(outcome.picked.iter().map(|p| p.row.variant));
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), candidates, "{outcome:?}");
        assert_eq!(
            outcome.rejected.len() + usize::from(outcome.picked.is_some()),
            candidates,
            "{outcome:?}"
        );
        (outcome, probed)
    }

    fn picked(outcome: &Outcome) -> Option<&'static str> {
        outcome.picked.as_ref().map(|p| p.row.variant)
    }

    /// Each rejected row as `variant: word`.
    fn rejected(outcome: &Outcome) -> Vec<String> {
        outcome
            .rejected
            .iter()
            .map(|r| format!("{}: {}", r.row.variant, r.why))
            .collect()
    }

    /// The pick over fixed machines: the fastest tier first, then the best rank. Flash-Next's rows
    /// rank UD-Q4_K_XL, UD-Q3_K_XL, fp4-local, nvfp4 on a GGUF-first class.
    #[test]
    fn the_pick_takes_the_fastest_tier_then_the_best_rank() {
        let class = CardClass::Ampere;
        let name = "qwen3.8-flash-next";
        let no_fit = |need| refused("no-fit", need);

        // A 24 GB card with 256 GB of host memory: nothing holds on the card alone.
        let (o, probed) = run(
            name,
            class,
            ROOM,
            &[],
            &[
                ("UD-Q4_K_XL", plan(Tier::Host)),
                ("UD-Q3_K_XL", plan(Tier::Host)),
                ("fp4-local", no_fit(102_300_000_000)),
                ("nvfp4", no_fit(132_700_000_000)),
            ],
        );
        assert_eq!(probed, ["UD-Q4_K_XL", "UD-Q3_K_XL", "fp4-local", "nvfp4"]);
        assert_eq!(picked(&o), Some("UD-Q4_K_XL"));
        assert_eq!(
            rejected(&o),
            [
                "UD-Q3_K_XL: ranked-below",
                "nvfp4: plan:no-fit",
                "fp4-local: plan:no-fit",
                "Q8_0: explicit-only"
            ]
        );
        assert_eq!(o.least_need_bytes(), Some(102_300_000_000));
        assert_eq!(o.picked.as_ref().map(|p| p.plan.tier()), Some(Tier::Host));

        // A row the cards hold whole beats a better-ranked one that needs the host, and ends the
        // asking: the rows below it are not planned.
        let (o, probed) = run(
            name,
            class,
            ROOM,
            &[],
            &[
                ("UD-Q4_K_XL", plan(Tier::Host)),
                ("UD-Q3_K_XL", plan(Tier::Cards)),
            ],
        );
        assert_eq!(probed, ["UD-Q4_K_XL", "UD-Q3_K_XL"]);
        assert_eq!(picked(&o), Some("UD-Q3_K_XL"));
        assert_eq!(
            rejected(&o),
            [
                "UD-Q4_K_XL: host",
                "nvfp4: ranked-below",
                "fp4-local: ranked-below",
                "Q8_0: explicit-only"
            ]
        );
        // The best-ranked row the cards hold is the only one asked, and asks nothing of the repos
        // of the rows below it.
        let mut listed: Vec<&str> = Vec::new();
        let target = CatalogRef::parse(name).expect("target");
        let machine = Machine {
            class,
            card_bytes: ROOM,
            cache_free_bytes: ROOM,
        };
        let o = pick(
            &target,
            &machine,
            &mut |row| {
                listed.push(row.variant);
                Avail::Present { download_bytes: 0 }
            },
            &mut |_| plan(Tier::Cards),
        );
        assert_eq!(listed, ["UD-Q4_K_XL"]);
        assert_eq!(picked(&o), Some("UD-Q4_K_XL"));
        assert_eq!(
            rejected(&o),
            [
                "UD-Q3_K_XL: ranked-below",
                "nvfp4: ranked-below",
                "fp4-local: ranked-below",
                "Q8_0: explicit-only"
            ]
        );

        // A row paged from NVMe loses to one the host's memory holds, and wins alone.
        let (o, _) = run(
            name,
            class,
            ROOM,
            &[],
            &[
                ("UD-Q4_K_XL", plan(Tier::Paged)),
                ("UD-Q3_K_XL", plan(Tier::Host)),
                ("fp4-local", no_fit(1)),
                ("nvfp4", no_fit(1)),
            ],
        );
        assert_eq!(picked(&o), Some("UD-Q3_K_XL"));
        assert_eq!(rejected(&o)[0], "UD-Q4_K_XL: paged");
        let (o, _) = run(
            name,
            class,
            ROOM,
            &[],
            &[
                ("UD-Q4_K_XL", plan(Tier::Paged)),
                ("UD-Q3_K_XL", plan(Tier::Paged)),
                ("fp4-local", no_fit(1)),
                ("nvfp4", no_fit(1)),
            ],
        );
        assert_eq!(picked(&o), Some("UD-Q4_K_XL"));
        assert_eq!(rejected(&o)[0], "UD-Q3_K_XL: ranked-below");

        // An 8 GB card with 16 GB of host memory: nothing holds, and the refusal names its need.
        let (o, _) = run(
            "qwen3-30b-a3b-instruct-2507",
            class,
            ROOM,
            &[],
            &[("Q4_K_M", no_fit(18_556_686_752))],
        );
        assert_eq!(picked(&o), None);
        assert_eq!(rejected(&o), ["Q4_K_M: plan:no-fit", "Q8_0: explicit-only"]);
        assert_eq!(o.least_need_bytes(), Some(18_556_686_752));
    }

    /// A row is not asked once it cannot beat the best plan in hand: after a plan the host holds,
    /// the rows whose files are larger than the cards' memory together (which cannot be held by the
    /// cards alone); after a plan paged from NVMe, none (a row the host holds beats it).
    #[test]
    fn a_row_too_large_for_the_cards_is_not_asked_once_the_host_holds_one() {
        let name = "qwen3.8-flash-next";
        let target = CatalogRef::parse(name).expect("target");
        let on = |card_bytes: u64, answers: &[(&str, Result<Plan, Refusal>)]| {
            let machine = Machine {
                class: CardClass::Ampere,
                card_bytes,
                cache_free_bytes: ROOM,
            };
            let mut asked: Vec<&str> = Vec::new();
            let outcome = pick(
                &target,
                &machine,
                &mut |_| Avail::Present { download_bytes: 0 },
                &mut |row| {
                    asked.push(row.variant);
                    answers
                        .iter()
                        .find(|(variant, _)| *variant == row.variant)
                        .map(|(_, answer)| answer.clone())
                        .unwrap_or_else(|| panic!("no probe answer for {}", row.variant))
                },
            );
            (outcome, asked)
        };
        // 48 GB of cards: every row is larger (the smallest, UD-Q3_K_XL, is 90 GB), so the first
        // plan the host holds is the pick and nothing below it is asked.
        let host = [("UD-Q4_K_XL", plan(Tier::Host))];
        let (o, asked) = on(48_000_000_000, &host);
        assert_eq!(asked, ["UD-Q4_K_XL"]);
        assert_eq!(picked(&o), Some("UD-Q4_K_XL"));
        assert_eq!(
            rejected(&o),
            [
                "UD-Q3_K_XL: ranked-below",
                "nvfp4: ranked-below",
                "fp4-local: ranked-below",
                "Q8_0: explicit-only"
            ]
        );
        // 100 GB of cards: UD-Q3_K_XL (90 GB) may be held by them, so it is asked, and a plan the
        // cards hold ends the asking.
        let both = [
            ("UD-Q4_K_XL", plan(Tier::Host)),
            ("UD-Q3_K_XL", plan(Tier::Cards)),
        ];
        let (o, asked) = on(100_000_000_000, &both);
        assert_eq!(asked, ["UD-Q4_K_XL", "UD-Q3_K_XL"]);
        assert_eq!(picked(&o), Some("UD-Q3_K_XL"));
        assert_eq!(rejected(&o)[0], "UD-Q4_K_XL: host");
        // A plan paged from NVMe is beaten by any row the host holds: the rows are all asked, down
        // to the first the host holds.
        let paged = [
            ("UD-Q4_K_XL", plan(Tier::Paged)),
            ("UD-Q3_K_XL", plan(Tier::Host)),
        ];
        let (o, asked) = on(48_000_000_000, &paged);
        assert_eq!(asked, ["UD-Q4_K_XL", "UD-Q3_K_XL"]);
        assert_eq!(picked(&o), Some("UD-Q3_K_XL"));
        assert_eq!(rejected(&o)[0], "UD-Q4_K_XL: paged");
    }

    /// The card class orders the formats: an FP4 folder first where the build has native FP4
    /// products (a GB10, a discrete 12.0 card), after every GGUF where it has none.
    #[test]
    fn the_card_class_orders_the_formats() {
        // Neither holds on the cards alone, so both are asked, in the class's order.
        let both = [("Q4_K_M", plan(Tier::Host)), ("nvfp4", plan(Tier::Host))];
        for (class, first, probed) in [
            (CardClass::Gb10, "nvfp4", ["nvfp4", "Q4_K_M"]),
            (CardClass::Blackwell, "nvfp4", ["nvfp4", "Q4_K_M"]),
            (CardClass::Ampere, "Q4_K_M", ["Q4_K_M", "nvfp4"]),
        ] {
            let (o, asked) = run("qwen3.8-27b", class, ROOM, &[], &both);
            assert_eq!(asked, probed, "{class:?}");
            assert_eq!(picked(&o), Some(first), "{class:?}");
        }
    }

    /// Every candidate of a pick is the pick or carries one word: a row nobody asked for, a row
    /// the repo or the disk cannot give, a row the plan refuses.
    #[test]
    fn every_row_is_picked_or_rejected_by_one_word() {
        let class = CardClass::Ampere;
        let held = [("UD-Q4_K_XL", plan(Tier::Host))];

        // The download is larger than the cache's room: the row is not probed at all.
        let short = [(
            "UD-Q4_K_XL",
            Avail::Present {
                download_bytes: 199_707_321_472,
            },
        )];
        let (o, probed) = run("glm-5.3-flash", class, 100_000_000_000, &short, &held);
        assert!(probed.is_empty());
        assert_eq!(picked(&o), None);
        assert_eq!(
            rejected(&o),
            ["UD-Q4_K_XL: disk-short", "Q8_0: explicit-only"]
        );
        // The same row already cached needs no room.
        let (o, _) = run("glm-5.3-flash", class, 0, &[], &held);
        assert_eq!(picked(&o), Some("UD-Q4_K_XL"));
        assert_eq!(o.picked.as_ref().map(|p| p.download_bytes), Some(0));

        // The listing lacks a set, or there is no listing and nothing cached.
        let listing = [
            ("UD-Q4_K_XL", Avail::NotInRepo),
            ("UD-Q3_K_XL", Avail::OfflineUncached),
        ];
        let (o, _) = run(
            "qwen3.8-flash-next",
            class,
            ROOM,
            &listing,
            &[
                ("fp4-local", plan(Tier::Cards)),
                ("nvfp4", plan(Tier::Cards)),
            ],
        );
        assert_eq!(picked(&o), Some("fp4-local"));
        assert_eq!(
            rejected(&o),
            [
                "UD-Q4_K_XL: not-in-repo",
                "UD-Q3_K_XL: offline-uncached",
                "nvfp4: ranked-below",
                "Q8_0: explicit-only"
            ]
        );

        // An unverified row is never taken on its own, and a start that names it takes it.
        let bolt = "nemotron-3.5-bolt-74b-a9b";
        let (o, probed) = run(bolt, class, ROOM, &[], &[]);
        assert!(probed.is_empty());
        assert_eq!(picked(&o), None);
        assert_eq!(rejected(&o), ["24GB: unverified"]);
        let (o, probed) = run(
            &format!("{bolt}:24gb"),
            class,
            ROOM,
            &[],
            &[("24GB", plan(Tier::Cards))],
        );
        assert_eq!(probed, ["24GB"]);
        assert_eq!(picked(&o), Some("24GB"));
        assert!(o.rejected.is_empty());
        assert!(o.picked.is_some_and(|p| !p.row.status.is_verified()));

        // An explicit-only format likewise.
        let (o, probed) = run(
            "qwen3.8-flash-next:q8_0",
            class,
            ROOM,
            &[],
            &[("Q8_0", plan(Tier::Host))],
        );
        assert_eq!(probed, ["Q8_0"]);
        assert_eq!(picked(&o), Some("Q8_0"));
    }

    /// A target is a model's name, alias or seat word and optionally one of its variants; an
    /// unknown name or variant is refused listing the known ones.
    #[test]
    fn a_target_parses_by_name_alias_seat_word_and_variant() {
        let by = |s: &str| CatalogRef::parse(s).unwrap_or_else(|e| panic!("{s}: {e}"));
        let glm = by("glm-5.3-flash");
        assert_eq!((glm.model.name, glm.variant), ("glm-5.3-flash", None));
        for same in ["GLM-5.3-Flash", "glm", "GLM"] {
            assert_eq!(by(same), glm, "{same}");
        }
        // A seat word that serves several models is the default one's.
        assert_eq!(by("qwen3").model.name, "qwen3-30b-a3b-instruct-2507");
        assert_eq!(by("minimaxm3").model.name, "minimax-m3.1-flash");
        let q = by("Qwen3.8-Flash-Next:ud-q3_k_xl");
        assert_eq!(
            (q.model.name, q.variant),
            ("qwen3.8-flash-next", Some("UD-Q3_K_XL"))
        );
        assert_eq!(q.to_string(), "qwen3.8-flash-next:UD-Q3_K_XL");

        let e = CatalogRef::parse("qwen9")
            .expect_err("unknown name")
            .to_string();
        for model in models() {
            assert!(e.contains(model.name), "{e}");
        }
        let e = CatalogRef::parse("glm:nope")
            .expect_err("unknown variant")
            .to_string();
        assert!(e.contains("UD-Q4_K_XL") && e.contains("Q8_0"), "{e}");
        for empty in ["", ":Q4_K_M", "glm:"] {
            assert!(
                matches!(CatalogRef::parse(empty), Err(CatalogError::Empty(_))),
                "{empty:?}"
            );
        }
    }

    /// A name or alias is one model's, and a seat word is the alias of exactly one model of the
    /// seat, so `--model <seat word>` resolves to one model.
    #[test]
    fn a_name_is_one_model_and_a_seat_word_names_its_default() {
        let mut spellings: Vec<String> = Vec::new();
        for model in models() {
            assert_eq!(
                model.name,
                model.name.to_ascii_lowercase(),
                "{}",
                model.name
            );
            spellings.push(model.name.to_owned());
            spellings.extend(model.aliases.iter().map(|a| a.to_ascii_lowercase()));
        }
        let total = spellings.len();
        spellings.sort_unstable();
        spellings.dedup();
        assert_eq!(spellings.len(), total, "a name or alias is two models'");
        for model in models() {
            let defaults: Vec<&str> = models()
                .iter()
                .filter(|m| m.seat == model.seat && m.aliases.contains(&model.seat))
                .map(|m| m.name)
                .collect();
            assert_eq!(defaults.len(), 1, "seat {}: {defaults:?}", model.seat);
        }
    }

    /// A verified row is pinned to a full commit; every row spells a repo `--hf` takes, a GGUF
    /// row's variant is its quant tag, and a model's variants are distinct.
    #[test]
    fn a_verified_row_is_pinned_and_every_row_spells_a_repo() {
        for model in models() {
            assert!(!model.rows.is_empty(), "{} has no row", model.name);
            let mut variants: Vec<String> = Vec::new();
            for row in model.rows {
                let at = format!("{} {}", model.name, row.variant);
                variants.push(row.variant.to_ascii_lowercase());
                if row.status.is_verified() {
                    let full = row.revision.len() == 40
                        && row
                            .revision
                            .bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
                    assert!(full, "{at}: a verified row pins a full commit");
                }
                assert!(!row.status.note().is_empty(), "{at}: no note");
                let parsed =
                    crate::RepoRef::parse(&row.hf()).unwrap_or_else(|e| panic!("{at}: {e}"));
                let quant = (row.kind == Kind::Gguf).then_some(row.variant);
                assert_eq!(parsed.quant.as_deref(), quant, "{at}");
            }
            let total = variants.len();
            variants.sort_unstable();
            variants.dedup();
            assert_eq!(variants.len(), total, "{}: a variant twice", model.name);
        }
    }

    /// Each class orders exactly the automatic formats, once each: a format missing from the order
    /// would never rank.
    #[test]
    fn every_class_orders_exactly_the_automatic_formats() {
        for class in [CardClass::Ampere, CardClass::Blackwell, CardClass::Gb10] {
            let formats = order(class);
            for format in Format::ALL {
                let places = formats.iter().filter(|f| **f == format).count();
                assert_eq!(
                    places,
                    usize::from(format.automatic()),
                    "{class:?} {format:?}"
                );
            }
        }
    }

    /// `tools/catalog.tsv` is the catalog's print.
    #[test]
    fn the_committed_print_is_the_catalog() {
        assert_eq!(
            read(CATALOG),
            print(),
            "tools/catalog.tsv is not the catalog's print; replace it with `hf::catalog::print()`"
        );
    }

    /// Every row names an architecture `tools/arch-registry.tsv` holds and a seat the serve binary
    /// has a word for.
    #[test]
    fn every_row_names_a_registry_architecture_and_a_served_seat() {
        let registry = read(ARCH_REGISTRY);
        let archs: Vec<&str> = registry
            .lines()
            .filter(|line| !line.starts_with('#'))
            .skip(1)
            .filter_map(|line| line.split('\t').next())
            .flat_map(|names| names.split(','))
            .collect();
        let serve = read(SERVE);
        for model in models() {
            assert!(
                archs.contains(&model.arch),
                "{}: {} is not in tools/arch-registry.tsv",
                model.name,
                model.arch
            );
            let word = format!("\"{}\" => Ok(Word::Seat(", model.seat);
            assert!(
                serve.contains(&word),
                "{}: {} is not a --model word of bloomery_serve.rs",
                model.name,
                model.seat
            );
        }
    }

    /// A GGUF row that the type census pins carries the census's revision and the sum of its set's
    /// shards.
    #[test]
    fn the_census_backed_rows_carry_its_revision_and_bytes() {
        let census = read(CENSUS);
        let lines: Vec<Vec<&str>> = census
            .lines()
            .skip(1)
            .map(|line| line.split('\t').collect())
            .collect();
        let mut backed = 0;
        for model in models() {
            for row in model.rows.iter().filter(|r| r.kind == Kind::Gguf) {
                let at = format!("{} {}", model.name, row.variant);
                // The set of the row: its name ends in the quant tag, a draft's sets left out.
                let shards: Vec<&Vec<&str>> = lines
                    .iter()
                    .filter(|f| format!("{}/{}", f[0], f[1]) == row.repo)
                    .filter(|f| {
                        let set = f[3];
                        let draft = ["mtp-", "dflash-", "eagle3-", "dspark-"]
                            .iter()
                            .any(|p| set.starts_with(p));
                        !draft && set.ends_with(&format!("-{}", row.variant))
                    })
                    .collect();
                if shards.is_empty() {
                    continue;
                }
                backed += 1;
                let mut sets: Vec<&str> = shards.iter().map(|f| f[3]).collect();
                sets.sort_unstable();
                sets.dedup();
                assert_eq!(sets.len(), 1, "{at}: the census has {sets:?}");
                assert!(
                    shards.iter().all(|f| f[2] == row.revision),
                    "{at}: revision"
                );
                let bytes: u64 = shards
                    .iter()
                    .map(|f| f[5].parse::<u64>().unwrap_or_else(|e| panic!("{at}: {e}")))
                    .sum();
                assert_eq!(row.bytes, Some(bytes), "{at}: bytes");
            }
        }
        assert!(backed >= 10, "only {backed} rows are held to the census");
    }

    /// A listing entry of `size` bytes whose digest is `digest`.
    fn entry(path: &str, size: u64, digest: &str) -> Entry {
        Entry {
            path: path.to_owned(),
            size,
            check: crate::Check::Sha256,
            digest: digest.to_owned(),
        }
    }

    fn row_of(model: &str, variant: &str) -> &'static Row {
        find(model)
            .unwrap_or_else(|e| panic!("{model}: {e}"))
            .row(variant)
            .unwrap_or_else(|| panic!("{model} has no {variant}"))
    }

    /// A row's files are the listing's set it names, the bytes still to fetch are the files the
    /// cache does not hold by path, size and digest, a listing without the set is `NotInRepo`,
    /// and offline only a whole verified set is present.
    #[test]
    fn availability_reads_the_listing_and_the_cache() {
        let row = row_of("qwen3", "Q4_K_M");
        let listed = [
            entry("Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf", 100, "aa"),
            entry("Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf", 200, "bb"),
            entry("README.md", 5, "cc"),
        ];
        assert_eq!(
            availability(row, &listed, &[]).expect("listed"),
            Avail::Present {
                download_bytes: 100
            }
        );
        let cached = [entry("Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf", 100, "aa")];
        assert_eq!(
            availability(row, &listed, &cached).expect("cached"),
            Avail::Present { download_bytes: 0 }
        );
        let moved = [entry("Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf", 100, "zz")];
        assert_eq!(
            availability(row, &listed, &moved).expect("a moved file"),
            Avail::Present {
                download_bytes: 100
            },
            "a cached file whose digest is not the listing's is fetched again"
        );
        assert_eq!(
            availability(row, &listed[1..], &[]).expect("no such set"),
            Avail::NotInRepo
        );
        assert_eq!(
            availability_offline(row, &cached),
            Avail::Present { download_bytes: 0 }
        );
        assert_eq!(availability_offline(row, &[]), Avail::OfflineUncached);

        let folder = row_of("qwen3.8-flash-next", "nvfp4");
        let checkpoint = [
            entry("model.safetensors.index.json", 10, "i"),
            entry("model-00001-of-00002.safetensors", 1000, "s1"),
            entry("model-00002-of-00002.safetensors", 2000, "s2"),
            entry("config.json", 3, "c"),
            entry("sub/other.json", 4, "o"),
        ];
        assert_eq!(
            availability(folder, &checkpoint, &checkpoint[..2]).expect("folder"),
            Avail::Present {
                download_bytes: 2003
            }
        );
        assert_eq!(
            availability(folder, &listed, &[]).expect("a repo with no index"),
            Avail::NotInRepo
        );
    }

    /// The next smaller variant is the largest verified automatic row under the row's bytes that
    /// the class ranks; an explicit-only or unverified row is never one, and the smallest row has
    /// none.
    #[test]
    fn the_next_smaller_variant_is_the_largest_one_under_the_row() {
        let model = find("qwen3.8-flash-next").expect("model");
        let q4 = model.row("UD-Q4_K_XL").expect("row");
        let q3 = model.row("UD-Q3_K_XL").expect("row");
        let nvfp4 = model.row("nvfp4").expect("row");
        let local = model.row("fp4-local").expect("row");
        let q8 = model.row("Q8_0").expect("row");
        assert_eq!(next_smaller(model, CardClass::Ampere, nvfp4), Some(q4));
        assert_eq!(next_smaller(model, CardClass::Ampere, q4), Some(local));
        assert_eq!(next_smaller(model, CardClass::Ampere, local), Some(q3));
        assert_eq!(next_smaller(model, CardClass::Ampere, q3), None);
        assert_eq!(
            next_smaller(model, CardClass::Ampere, q8),
            Some(nvfp4),
            "the explicit-only row's next smaller is the largest automatic row"
        );
    }

    /// `--plan --json`'s answer is the last schema line of the output, whatever record lines
    /// share it; a refusal carries its code and need; a line that is not the contract's is named.
    #[test]
    fn a_plan_line_is_read_from_the_output() {
        let out = [
            "plan a card=A6000 ctx_max=4096",
            r#"{"schema":"bloomery.event/1","event":"error"}"#,
            r#"{"schema":"bloomery.plan/1","seat":"glm","place":"a","place_why":"w","ctx":8192,"ctx_rule":"fitted","slots":2,"limit":16384,"host_bytes":5,"paged":3,"need":null}"#,
        ]
        .join("\n");
        assert_eq!(
            read_plan(&out).expect("a plan"),
            Ok(Plan {
                place: "a".to_owned(),
                ctx: 8192,
                slots: 2,
                host_bytes: 5,
                paged_bytes: 3,
                fit: Fit::Planned,
            })
        );
        let refused =
            r#"{"schema":"bloomery.plan/1","refused":{"code":"no-fit","message":"m","need":77}}"#;
        assert_eq!(
            read_plan(refused).expect("a refusal"),
            Err(Refusal {
                code: "no-fit".to_owned(),
                need_bytes: Some(77)
            })
        );
        let e = read_plan("plan a\nnot json").expect_err("no schema line");
        assert!(e.contains(PLAN_SCHEMA), "{e}");
        let e = read_plan(r#"{"schema":"bloomery.plan/1","place":"a","ctx":1}"#)
            .expect_err("a plan without its slots");
        assert!(e.contains("slots"), "{e}");
    }

    /// The model's JSON entry names the pick with its fit and every rejection with its word.
    #[test]
    fn a_model_entry_names_the_pick_and_each_rejection() {
        let model = find("qwen3.8-flash-next").expect("model");
        let q4 = model.row("UD-Q4_K_XL").expect("row");
        let q3 = model.row("UD-Q3_K_XL").expect("row");
        let outcome = Outcome {
            picked: Some(Picked {
                row: q4,
                plan: plan(Tier::Cards).expect("held"),
                download_bytes: 0,
            }),
            rejected: vec![Rejected {
                row: q3,
                why: Reason::RankedBelow,
            }],
        };
        let v = model_json(model, &outcome);
        assert_eq!(v["name"], "qwen3.8-flash-next");
        assert_eq!(v["seat"], "qwen38");
        assert_eq!(v["pick"]["variant"], "UD-Q4_K_XL");
        assert_eq!(v["pick"]["source"], q4.hf());
        assert_eq!(v["pick"]["cached"], true);
        assert_eq!(v["pick"]["fit"], "planned");
        assert_eq!(v["rejected"][0]["variant"], "UD-Q3_K_XL");
        assert_eq!(v["rejected"][0]["why"], "ranked-below");
        let none = Outcome {
            picked: None,
            rejected: Vec::new(),
        };
        assert!(model_json(model, &none)["pick"].is_null());
    }
}
