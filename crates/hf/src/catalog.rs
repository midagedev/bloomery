//! The model catalog: the models bloomery starts by name, and for each one the files it can start
//! from. A row is a file of a Hugging Face repo at a pinned revision: a GGUF set (a quant tag of
//! the repo) or a safetensors checkpoint folder (the repo's one model, served as stored).
//!
//! The catalog is data and one rule. [`pick`] ranks a model's rows for a card class
//! ([`CardClass::order`]), leaves out what a start by name never takes on its own (a format that
//! is explicit-only, a row nobody has verified), and asks its caller for what only the machine can
//! say: whether the repo holds the row and how much is left to download ([`Avail`]), and whether
//! the seat's plan holds ([`Plan`]). The best-ranked row at the fastest placement tier wins: a plan
//! the cards hold whole, then one that needs the host's memory, then one paged from NVMe.
//!
//! `tools/catalog.tsv` is [`print`]'s committed copy.

mod rows;

use std::cmp::Reverse;
use std::fmt;

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

/// The kind of card the stage runs on, which decides the order of the formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CardClass {
    /// Compute capability 8.x, and a 12.0 card in a build without native FP4 products.
    Ampere,
    /// A discrete compute capability 12.0 card in a build with native FP4 products.
    Blackwell,
    /// The integrated compute capability 12.1 part (GB10) in a build with native FP4 products.
    Gb10,
}

impl CardClass {
    /// The spelling the JSON contract uses.
    pub const fn name(self) -> &'static str {
        match self {
            CardClass::Ampere => "ampere",
            CardClass::Blackwell => "blackwell",
            CardClass::Gb10 => "gb10",
        }
    }

    /// The automatic formats, best first; a row whose format is not here is never ranked.
    pub const fn order(self) -> &'static [Format] {
        match self {
            CardClass::Ampere => GGUF_FIRST,
            CardClass::Blackwell | CardClass::Gb10 => FP4_FIRST,
        }
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

/// Why a catalog target, or a pick over it, was refused.
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
    #[error("the probe answered {got} of the {asked} rows it was asked about")]
    ProbeAnswers { asked: usize, got: usize },
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

/// What a pick reads of the machine: the card class that orders the formats and the room the
/// download cache has.
#[derive(Clone, Copy, Debug)]
pub struct Machine {
    pub class: CardClass,
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
    /// A row of the same or a faster tier ranks above it.
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
/// the unverified rows. `avail` says what the listing and the cache hold of a row (asked of the
/// rows that pass that filter), `probe` plans every surviving row at once, in rank order, and
/// answers each in the same order.
pub fn pick(
    target: &CatalogRef,
    machine: &Machine,
    avail: &mut dyn FnMut(&'static Row) -> Avail,
    probe: &mut dyn FnMut(&[&'static Row]) -> Vec<Result<Plan, Refusal>>,
) -> Result<Outcome, CatalogError> {
    let model = target.model;
    let named = target.variant.is_some();
    let mut order: Vec<&'static Row> = model
        .rows
        .iter()
        .filter(|row| target.variant.is_none_or(|variant| row.variant == variant))
        .collect();
    order.sort_by_key(|row| (rank(machine.class, row), Reverse(row.bytes.unwrap_or(0))));

    let mut rejected: Vec<Rejected> = Vec::new();
    let mut survivors: Vec<(&'static Row, u64)> = Vec::new();
    for row in order {
        match screen(row, named, machine, avail) {
            Ok(download_bytes) => survivors.push((row, download_bytes)),
            Err(why) => rejected.push(Rejected { row, why }),
        }
    }

    let asked: Vec<&'static Row> = survivors.iter().map(|&(row, _)| row).collect();
    let answers = if asked.is_empty() {
        Vec::new()
    } else {
        probe(&asked)
    };
    if answers.len() != asked.len() {
        return Err(CatalogError::ProbeAnswers {
            asked: asked.len(),
            got: answers.len(),
        });
    }
    let mut held: Vec<(&'static Row, u64, Plan)> = Vec::new();
    for ((row, download_bytes), answer) in survivors.into_iter().zip(answers) {
        match answer {
            Ok(plan) => held.push((row, download_bytes, plan)),
            Err(refusal) => rejected.push(Rejected {
                row,
                why: Reason::Plan {
                    code: refusal.code,
                    need_bytes: refusal.need_bytes,
                },
            }),
        }
    }

    let best = held.iter().map(|(_, _, plan)| plan.tier()).min();
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
    Ok(Outcome { picked, rejected })
}

/// The row's place in the class's order; a format the order does not hold ranks last.
fn rank(class: CardClass, row: &Row) -> usize {
    class
        .order()
        .iter()
        .position(|format| *format == row.format)
        .unwrap_or(usize::MAX)
}

/// The bytes still to fetch of a row that may be probed, or the reason it may not.
fn screen(
    row: &'static Row,
    named: bool,
    machine: &Machine,
    avail: &mut dyn FnMut(&'static Row) -> Avail,
) -> Result<u64, Reason> {
    if !named {
        if !row.format.automatic() {
            return Err(Reason::ExplicitOnly);
        }
        if !row.status.is_verified() {
            return Err(Reason::Unverified);
        }
    }
    match avail(row) {
        Avail::NotInRepo => Err(Reason::NotInRepo),
        Avail::OfflineUncached => Err(Reason::OfflineUncached),
        Avail::Present { download_bytes } if download_bytes > machine.cache_free_bytes => {
            Err(Reason::DiskShort)
        }
        Avail::Present { download_bytes } => Ok(download_bytes),
    }
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
            &mut |rows| {
                rows.iter()
                    .map(|row| {
                        probed.push(row.variant);
                        answers
                            .iter()
                            .find(|(variant, _)| *variant == row.variant)
                            .map(|(_, answer)| answer.clone())
                            .unwrap_or_else(|| panic!("no probe answer for {}", row.variant))
                    })
                    .collect()
            },
        )
        .expect("the probe answers every row");
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

        // A row the cards hold whole beats a better-ranked one that needs the host.
        let (o, _) = run(
            name,
            class,
            ROOM,
            &[],
            &[
                ("UD-Q4_K_XL", plan(Tier::Host)),
                ("UD-Q3_K_XL", plan(Tier::Cards)),
                ("fp4-local", no_fit(1)),
                ("nvfp4", no_fit(1)),
            ],
        );
        assert_eq!(picked(&o), Some("UD-Q3_K_XL"));
        assert_eq!(
            rejected(&o),
            [
                "UD-Q4_K_XL: host",
                "nvfp4: plan:no-fit",
                "fp4-local: plan:no-fit",
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

    /// The card class orders the formats: an FP4 folder first where the build has native FP4
    /// products (a GB10, a discrete 12.0 card), after every GGUF where it has none.
    #[test]
    fn the_card_class_orders_the_formats() {
        let both = [("Q4_K_M", plan(Tier::Cards)), ("nvfp4", plan(Tier::Cards))];
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

    #[test]
    fn a_probe_that_answers_the_wrong_number_of_rows_is_refused() {
        let target = CatalogRef::parse("glm").expect("glm");
        let machine = Machine {
            class: CardClass::Ampere,
            cache_free_bytes: ROOM,
        };
        let e = pick(
            &target,
            &machine,
            &mut |_| Avail::Present { download_bytes: 0 },
            &mut |_| Vec::new(),
        )
        .expect_err("no answers");
        assert!(matches!(e, CatalogError::ProbeAnswers { asked: 1, got: 0 }));
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
            let order = class.order();
            for format in Format::ALL {
                let places = order.iter().filter(|f| **f == format).count();
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
}
