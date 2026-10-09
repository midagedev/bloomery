//! The fixture tier: which file a gate runs on, and which of its clauses that
//! file can answer.
//!
//! `BLOOMERY_TIER` is `real` (unset means the same: the files the model
//! profiles name) or `fixture` (the small files `fixture generate` writes,
//! one directory a family; `tools/ref/ref-paths.sh` resolves the file and
//! `tools/box.sh` exports it as `BLOOMERY_REF_MODEL`). This module owns the
//! one read of the variable in the crates, the rule of which clauses a tier
//! runs, and the check that a fixture-tier run opened a fixture.
//!
//! A clause is one verdict a gate prints. It carries exactly one [`Tag`]:
//!
//! - [`Tag::SelfConsistency`]: two arms of the engine under one plan agree
//!   (eager and replay, step and prefill, verify and plain). The weights do
//!   not matter, so a fixture answers it.
//! - [`Tag::Oracle`]: a comparison with a reference engine's output on the
//!   real file.
//! - [`Tag::FileBound`]: a clause that reads the real file's content (its
//!   text behaviour, its tokenizer ids, an accepted-draft count).
//! - [`Tag::Scale`]: a clause that needs the real shapes (node counts,
//!   queue saturation, capacity, page faults).
//! - [`Tag::FixtureOracle`]: a comparison with a reference engine's output on
//!   the fixture file (ik's sets dumped on that fixture). It opens its set
//!   through [`fixture_set`].
//!
//! The `real` tier runs every clause except the fixture's oracle comparisons,
//! tagged or not: nothing changes for a gate that declares no tags. Each of
//! those is a [`FIXTURE_ONLY`] line. The `fixture` tier runs the
//! self-consistency clauses and the fixture's oracle comparisons (each a
//! [`FIXTURE_ORACLE`] line) and defers the other three, each deferral a
//! [`DEFERRED`] line; the batch runner (`tools/gate-batch.sh`) counts the
//! deferral lines. A clause with no tag in the fixture tier is a named error,
//! never a clause that quietly ran or quietly did not.

use std::fmt::{self, Debug};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use bloomery_levers::{CARD_BUDGET, Levers};
use gguf::Split;
use model::fixture::KEY_CARD_BUDGET;
use model::placement::PlanLevers;
use refset::RefError;
use refset::family::{Family, Identity};

use crate::{GateError, verdict};

/// The variable the tier is read from.
pub const ENV: &str = "BLOOMERY_TIER";

/// A line that starts with this is a clause the fixture tier left to the real
/// tier. `tools/gate-batch.sh` counts these lines per item.
pub const DEFERRED: &str = "deferred(real)";

/// A line that starts with this is a [`Tag::FixtureOracle`] clause the real
/// tier left to the fixture tier.
pub const FIXTURE_ONLY: &str = "fixture-only(oracle)";

/// A line that starts with this is a [`Tag::FixtureOracle`] clause the fixture
/// tier ran.
pub const FIXTURE_ORACLE: &str = "fixture-oracle";

/// The key the generator writes into every fixture's header.
pub const VERSION_KEY: &str = "bloomery.fixture.version";

/// The key the generator writes into a file that holds only some of the
/// planned tensors; the engine must not run such a file.
pub const SUBSET_KEY: &str = "bloomery.fixture.subset";

/// Which tier a gate runs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// The files the model profiles name; every clause runs but the
    /// fixture's oracle comparisons.
    Real,
    /// A generated small file; the self-consistency clauses and the fixture's
    /// oracle comparisons run.
    Fixture,
}

/// What a clause answers (the module doc names the five).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tag {
    /// Two arms of the engine under one plan agree.
    SelfConsistency,
    /// A comparison with a reference engine's output on the real file.
    Oracle,
    /// A clause bound to the real file's content.
    FileBound,
    /// A clause that needs the real shapes.
    Scale,
    /// A comparison with a reference engine's output on the fixture file (ik's
    /// sets dumped on that fixture).
    FixtureOracle,
}

impl Tag {
    /// Every tag, in the order the module doc names them.
    pub const ALL: [Tag; 5] = [
        Tag::SelfConsistency,
        Tag::Oracle,
        Tag::FileBound,
        Tag::Scale,
        Tag::FixtureOracle,
    ];

    /// The tag's name as a [`DEFERRED`] line prints it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Tag::SelfConsistency => "self-consistency",
            Tag::Oracle => "oracle",
            Tag::FileBound => "file-bound",
            Tag::Scale => "scale",
            Tag::FixtureOracle => "fixture-oracle",
        }
    }
}

/// What a tier does with one clause.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    /// The gate runs the clause.
    Run,
    /// The clause is left to the real tier.
    Deferred(Tag),
    /// The clause is a [`Tag::FixtureOracle`] one, which only the fixture
    /// tier can answer.
    FixtureOnly,
}

impl Decision {
    /// The line a gate prints for this decision of `clause`, which declares
    /// `tag`: the clause's [`DEFERRED`] line, its [`FIXTURE_ONLY`] line, or
    /// its [`FIXTURE_ORACLE`] line when the fixture tier runs it; `None` for a
    /// clause that runs without one.
    #[must_use]
    pub fn line(self, clause: &str, tag: Option<Tag>) -> Option<String> {
        match (self, tag) {
            (Decision::Run, Some(Tag::FixtureOracle)) => Some(fixture_oracle_line(clause)),
            (Decision::Run, _) => None,
            (Decision::Deferred(tag), _) => Some(deferred_line(clause, tag)),
            (Decision::FixtureOnly, _) => Some(fixture_only_line(clause)),
        }
    }
}

/// Why the tier refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TierError {
    /// `BLOOMERY_TIER` is neither `real` nor `fixture`.
    BadValue(String),
    /// A clause with no tag in the fixture tier.
    Untagged(String),
    /// The fixture tier opened a file whose header has no
    /// [`VERSION_KEY`]: a real file, or one this generator did not write.
    NotFixture(String),
    /// The fixture tier opened a file that holds only a subset of the
    /// planned tensors ([`SUBSET_KEY`]).
    SubsetFile(String),
}

impl fmt::Display for TierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TierError::BadValue(v) => {
                write!(f, "{ENV} is real or fixture (unset: real), got {v:?}")
            }
            TierError::Untagged(clause) => write!(
                f,
                "the clause {clause:?} declares no tag: under {ENV}=fixture every clause is one of \
                 {} (crates/gpu-gates/src/tier.rs)",
                Tag::ALL.map(Tag::name).join(", ")
            ),
            TierError::NotFixture(path) => write!(
                f,
                "{ENV}=fixture, but {path} has no {VERSION_KEY} key: it is not a fixture file, and \
                 the fixture tier never opens a real one"
            ),
            TierError::SubsetFile(path) => write!(
                f,
                "{ENV}=fixture, but {path} carries {SUBSET_KEY}: it holds only some of the planned \
                 tensors, and the engine must not run it"
            ),
        }
    }
}

impl std::error::Error for TierError {}

impl Tier {
    /// The tier a value of `BLOOMERY_TIER` names; unset and empty are
    /// [`Tier::Real`], as in the shell scripts (`${BLOOMERY_TIER:-real}`).
    ///
    /// # Errors
    /// [`TierError::BadValue`] for any other value.
    pub fn parse(value: Option<&str>) -> Result<Tier, TierError> {
        match value {
            None | Some("" | "real") => Ok(Tier::Real),
            Some("fixture") => Ok(Tier::Fixture),
            Some(other) => Err(TierError::BadValue(other.to_string())),
        }
    }

    /// The tier `BLOOMERY_TIER` names in this process.
    ///
    /// # Errors
    /// [`TierError::BadValue`] for a value that is not UTF-8 or not a tier.
    pub fn from_env() -> Result<Tier, TierError> {
        match std::env::var_os(ENV) {
            None => Ok(Tier::Real),
            Some(v) => match v.to_str() {
                Some(s) => Tier::parse(Some(s)),
                None => Err(TierError::BadValue(v.to_string_lossy().into_owned())),
            },
        }
    }

    /// The tier's name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Tier::Real => "real",
            Tier::Fixture => "fixture",
        }
    }

    /// Whether this tier runs a clause carrying `tag`.
    #[must_use]
    pub fn runs(self, tag: Tag) -> bool {
        match self {
            Tier::Real => tag != Tag::FixtureOracle,
            Tier::Fixture => matches!(tag, Tag::SelfConsistency | Tag::FixtureOracle),
        }
    }

    /// What this tier does with `clause`, which declares `tag` (`None`: no
    /// tag). The real tier runs every clause but a fixture-oracle one, which
    /// it leaves to the fixture tier. The fixture tier runs a self-consistency
    /// or fixture-oracle clause, defers the others, and refuses an untagged
    /// one.
    ///
    /// # Errors
    /// [`TierError::Untagged`] for a clause with no tag in the fixture tier.
    pub fn decide(self, clause: &str, tag: Option<Tag>) -> Result<Decision, TierError> {
        match (self, tag) {
            (Tier::Fixture, None) => Err(TierError::Untagged(clause.to_string())),
            (_, None) => Ok(Decision::Run),
            (_, Some(tag)) if self.runs(tag) => Ok(Decision::Run),
            // `runs` leaves only a fixture-oracle clause to the fixture tier.
            (Tier::Real, Some(_)) => Ok(Decision::FixtureOnly),
            (Tier::Fixture, Some(tag)) => Ok(Decision::Deferred(tag)),
        }
    }

    /// The gate's call for one clause: `Ok(true)` when it runs the clause,
    /// `Ok(false)` after printing the clause's [`DEFERRED`] or [`FIXTURE_ONLY`]
    /// line. A fixture-oracle clause the fixture tier runs prints its
    /// [`FIXTURE_ORACLE`] line first.
    ///
    /// # Errors
    /// As [`Tier::decide`].
    pub fn clause(self, clause: &str, tag: Option<Tag>) -> Result<bool, TierError> {
        let decision = self.decide(clause, tag)?;
        if let Some(line) = decision.line(clause, tag) {
            println!("{line}");
        }
        Ok(decision == Decision::Run)
    }

    /// The check that a run of this tier opened the kind of file it names,
    /// from the header's keys. The real tier looks at nothing; the fixture
    /// tier wants [`VERSION_KEY`] and refuses [`SUBSET_KEY`].
    ///
    /// # Errors
    /// [`TierError::NotFixture`] or [`TierError::SubsetFile`].
    pub fn check_file<'a>(
        self,
        path: &Path,
        keys: impl IntoIterator<Item = &'a str>,
    ) -> Result<(), TierError> {
        if self == Tier::Real {
            return Ok(());
        }
        let (mut version, mut subset) = (false, false);
        for key in keys {
            version |= key == VERSION_KEY;
            subset |= key == SUBSET_KEY;
        }
        let shown = path.display().to_string();
        if !version {
            Err(TierError::NotFixture(shown))
        } else if subset {
            Err(TierError::SubsetFile(shown))
        } else {
            Ok(())
        }
    }
}

/// The line a fixture-tier gate prints for a clause it leaves to the real
/// tier: [`DEFERRED`], the tag, the clause.
#[must_use]
pub fn deferred_line(clause: &str, tag: Tag) -> String {
    format!("{DEFERRED} {}: {clause}", tag.name())
}

/// The line a real-tier gate prints for a [`Tag::FixtureOracle`] clause it
/// leaves to the fixture tier: [`FIXTURE_ONLY`], the clause.
#[must_use]
pub fn fixture_only_line(clause: &str) -> String {
    format!("{FIXTURE_ONLY}: {clause}")
}

/// The line a fixture-tier gate prints when it runs a [`Tag::FixtureOracle`]
/// clause: [`FIXTURE_ORACLE`], the clause.
#[must_use]
pub fn fixture_oracle_line(clause: &str) -> String {
    format!("{FIXTURE_ORACLE}: {clause}")
}

/// How many clauses were decided to run, to be left to the real tier and to
/// be left to the fixture tier.
struct Tally {
    ran: AtomicUsize,
    left_to_real: AtomicUsize,
    left_to_fixture: AtomicUsize,
}

impl Tally {
    const fn new() -> Tally {
        Tally {
            ran: AtomicUsize::new(0),
            left_to_real: AtomicUsize::new(0),
            left_to_fixture: AtomicUsize::new(0),
        }
    }

    fn record(&self, decision: Decision) {
        let counter = match decision {
            Decision::Run => &self.ran,
            Decision::Deferred(_) => &self.left_to_real,
            Decision::FixtureOnly => &self.left_to_fixture,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// `(ran, left to the real tier, left to the fixture tier)`.
    fn counts(&self) -> (usize, usize, usize) {
        (
            self.ran.load(Ordering::Relaxed),
            self.left_to_real.load(Ordering::Relaxed),
            self.left_to_fixture.load(Ordering::Relaxed),
        )
    }
}

static TALLY: Tally = Tally::new();

/// Whether this process's tier runs the clause `name`, which carries `tag`; when it does not, the
/// clause's [`DEFERRED`] or [`FIXTURE_ONLY`] line is printed (the batch counts the first), and a
/// fixture-oracle clause the fixture tier runs prints its [`FIXTURE_ORACLE`] line. Every verdict a
/// gate prints is decided by one of these calls (the clause's function, or the sub-assertion that
/// compares with a reference), so a clause with no tag has no way to run: the tag is the call's
/// argument.
///
/// # Errors
/// [`TierError::BadValue`] for a `BLOOMERY_TIER` that is neither tier.
pub fn run_clause(name: &str, tag: Tag) -> Result<bool, GateError> {
    run_clause_in(
        Tier::from_env()?,
        name,
        tag,
        &TALLY,
        &mut std::io::stdout().lock(),
    )
}

/// [`run_clause`] of its inputs: the tier, the tally the decision is counted in and the sink its
/// line is written to.
fn run_clause_in(
    tier: Tier,
    name: &str,
    tag: Tag,
    tally: &Tally,
    out: &mut impl std::io::Write,
) -> Result<bool, GateError> {
    let decision = tier.decide(name, Some(tag))?;
    if let Some(line) = decision.line(name, Some(tag)) {
        writeln!(out, "{line}")?;
    }
    tally.record(decision);
    Ok(decision == Decision::Run)
}

/// A precondition that proves a clause is not vacuous (a flip landed, a tier slot was hit, a draft
/// token was accepted) and whose truth is a property of the file's content: `holds` is required
/// where the tier runs the premise, and not asserted where it defers (the fixture's uniform
/// routing reads 0 there, which proves nothing about the engine). The equality the premise guards
/// runs in both tiers.
///
/// # Errors
/// As [`run_clause`].
pub fn premise(name: &str, tag: Tag, holds: bool) -> Result<bool, GateError> {
    Ok(!run_clause(name, tag)? || holds)
}

/// A [`Tag::FileBound`] premise several clauses share, decided once per process: the first call
/// runs the clause (printing its [`DEFERRED`] line when this tier leaves it to the real one) and
/// caches the answer; every later call returns it without printing, so the premise counts once.
///
/// # Errors
/// As [`run_clause`].
pub fn premise_once(name: &'static str) -> Result<bool, GateError> {
    static SEEN: std::sync::Mutex<Vec<(&'static str, bool)>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().expect("the premise cache's lock");
    if let Some(&(_, runs)) = seen.iter().find(|(n, _)| *n == name) {
        return Ok(runs);
    }
    let runs = run_clause(name, Tag::FileBound)?;
    seen.push((name, runs));
    Ok(runs)
}

/// A self-consistency clause: it runs in both tiers, so a tier that deferred it is refused by name
/// here, never skipped.
///
/// # Errors
/// As [`run_clause`], or a tier that deferred a self-consistency clause.
pub fn sc(name: &str) -> Result<(), GateError> {
    if run_clause(name, Tag::SelfConsistency)? {
        Ok(())
    } else {
        Err(format!("the self-consistency clause {name:?} was deferred").into())
    }
}

/// Why a [`Tag::FixtureOracle`] clause could not open its set. Each case fails the gate by name;
/// none defers the clause.
#[derive(Debug)]
pub enum FixtureSetError {
    /// The family is not a fixture family: a real family's check cannot tell a set dumped from
    /// another generation of the fixture, so it would open one.
    NotFixtureFamily {
        clause: String,
        family: &'static str,
    },
    /// The set is not one of the family's.
    NotInFamily {
        clause: String,
        family: &'static str,
        set: String,
    },
    /// The set is not in place.
    Missing {
        clause: String,
        path: PathBuf,
        recipe: &'static str,
    },
    /// The family's check refused the set (stale, foreign, unfinished, malformed), or refused to
    /// name the fixture file it checks the set against.
    Refused {
        clause: String,
        error: Box<RefError>,
        recipe: &'static str,
    },
}

impl FixtureSetError {
    /// The family's refusal, for a set it refused.
    #[must_use]
    pub fn cause(&self) -> Option<&RefError> {
        match self {
            FixtureSetError::Refused { error, .. } => Some(error.as_ref()),
            _ => None,
        }
    }
}

impl fmt::Display for FixtureSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FixtureSetError::NotFixtureFamily { clause, family } => write!(
                f,
                "fixture oracle {clause}: the {family} family is not a fixture family \
                 (refset's fx- families hold the sets dumped on a fixture)"
            ),
            FixtureSetError::NotInFamily {
                clause,
                family,
                set,
            } => write!(
                f,
                "fixture oracle {clause}: {set} is not a set of the {family} family"
            ),
            FixtureSetError::Missing {
                clause,
                path,
                recipe,
            } => write!(
                f,
                "fixture oracle {clause}: set {} missing; dump it with `{recipe}`",
                path.display()
            ),
            FixtureSetError::Refused {
                clause,
                error,
                recipe,
            } => match error.as_ref() {
                RefError::Stale { .. } => {
                    write!(
                        f,
                        "fixture oracle {clause}: stale set: {error}; dump it again with `{recipe}`"
                    )
                }
                RefError::Foreign { .. } => {
                    write!(
                        f,
                        "fixture oracle {clause}: foreign set: {error}; dump it again with `{recipe}`"
                    )
                }
                RefError::Unfinished { .. } => {
                    write!(
                        f,
                        "fixture oracle {clause}: unfinished set: {error}; dump it again with `{recipe}`"
                    )
                }
                RefError::Malformed { .. } => {
                    write!(f, "fixture oracle {clause}: malformed set: {error}")
                }
                RefError::Missing { .. } => {
                    write!(
                        f,
                        "fixture oracle {clause}: the fixture file is missing: {error}"
                    )
                }
            },
        }
    }
}

impl std::error::Error for FixtureSetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FixtureSetError::Refused { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// The set `set` of fixture `family` that the [`Tag::FixtureOracle`] clause `clause` compares
/// against, under [`crate::data_dir`] and checked by the family: the set's path, which the clause's
/// reader opens. See [`fixture_set_in`].
///
/// # Errors
/// As [`fixture_set_in`].
pub fn fixture_set(clause: &str, family: &Family, set: &str) -> Result<PathBuf, GateError> {
    Ok(fixture_set_in(&crate::data_dir(), clause, family, set)?)
}

/// [`fixture_set`] under `data_dir`. The set is resolved as the family's [`Family::path`] does,
/// then opened through the family's check ([`Family::check_set`]): the file it was dumped from,
/// the generation of that file (`# fixture`), the ik build and the completion trailer. A set that
/// is not there, or that the check refuses, ends the gate; it never defers the clause.
///
/// # Errors
/// [`FixtureSetError`]: the family is not a fixture family, the set is not the family's, it is not
/// in place, or the family refused it.
pub fn fixture_set_in(
    data_dir: &Path,
    clause: &str,
    family: &Family,
    set: &str,
) -> Result<PathBuf, FixtureSetError> {
    let clause = clause.to_string();
    if !matches!(
        family.identity,
        Identity::FixtureManifest | Identity::FixtureMtpManifest
    ) {
        return Err(FixtureSetError::NotFixtureFamily {
            clause,
            family: family.name,
        });
    }
    if !family.sets.contains(&set) {
        return Err(FixtureSetError::NotInFamily {
            clause,
            family: family.name,
            set: set.to_string(),
        });
    }
    let path = data_dir.join(family.resolve.map_or_else(|| set.to_string(), |f| f(set)));
    let recipe = family.recipe;
    if !path.exists() {
        return Err(FixtureSetError::Missing {
            clause,
            path,
            recipe,
        });
    }
    match family.check_set(&path) {
        Ok(_) => Ok(path),
        Err(RefError::Missing { path: at, .. }) if at.starts_with(&path) => {
            Err(FixtureSetError::Missing {
                clause,
                path,
                recipe,
            })
        }
        Err(error) => Err(FixtureSetError::Refused {
            clause,
            error: Box::new(error),
            recipe,
        }),
    }
}

/// The levers a gate parses (`bloomery_levers::at_main`): `real`, and in the fixture tier the card
/// budget too — the fixture's runner exports the header's budget, which [`plan_levers`] then holds
/// equal to the header's.
///
/// # Errors
/// A `BLOOMERY_TIER` that is neither tier.
pub fn acts_on(real: &[&'static str]) -> Result<Vec<&'static str>, GateError> {
    Ok(acts_in(Tier::from_env()?, real))
}

/// [`acts_on`] of its two inputs: the tier and the real tier's levers.
#[must_use]
pub fn acts_in(tier: Tier, real: &[&'static str]) -> Vec<&'static str> {
    let mut levers = real.to_vec();
    if tier == Tier::Fixture && !levers.contains(&CARD_BUDGET) {
        levers.push(CARD_BUDGET);
    }
    levers
}

/// The clauses decided so far in this process: how many ran and how many were left to the real
/// tier, as the gate's closing line prints them ([`fixture_only`] counts the third kind).
#[must_use]
pub fn tally() -> (usize, usize) {
    let (ran, left, _) = TALLY.counts();
    (ran, left)
}

/// The fixture-oracle clauses this process's real tier left to the fixture tier.
#[must_use]
pub fn fixture_only() -> usize {
    TALLY.counts().2
}

/// The gate's closing line of the tally: `clauses: N ran, M left to the real tier`, and
/// `, K left to the fixture tier` after it when K is not 0.
#[must_use]
pub fn tally_line() -> String {
    let (ran, left, fixture) = TALLY.counts();
    tally_text(ran, left, fixture)
}

/// [`tally_line`] of its three counts.
#[must_use]
pub fn tally_text(ran: usize, left_to_real: usize, left_to_fixture: usize) -> String {
    let mut line = format!("clauses: {ran} ran, {left_to_real} left to the real tier");
    if left_to_fixture > 0 {
        line.push_str(&format!(", {left_to_fixture} left to the fixture tier"));
    }
    line
}

/// The move proof of one derived value: in the real tier the value beside the literal it replaced,
/// and whether they are equal; in the fixture tier the derived value beside the real file's, with
/// no verdict (the fixture's shape differs by construction).
pub fn witness<T: PartialEq + Debug>(name: &str, derived: T, real: T) -> bool {
    match Tier::from_env() {
        Ok(Tier::Fixture) => {
            println!("derived {name} = {derived:?} (the real file's {real:?})");
            true
        }
        _ => {
            let ok = derived == real;
            println!(
                "move proof {name}: derived {derived:?}, the old literal {real:?} {}",
                verdict(ok)
            );
            ok
        }
    }
}

/// The key the generator records the single-card budget under, read from a header.
///
/// # Errors
/// A key of another type than u64.
pub fn header_budget(split: &Split) -> Result<Option<u64>, GateError> {
    match split.value(KEY_CARD_BUDGET) {
        None => Ok(None),
        Some(v) => v
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("{KEY_CARD_BUDGET} holds {v:?}, not a u64").into()),
    }
}

/// The plan's levers: the real tier's are the caller's (`BLOOMERY_CARD_BUDGET`, unset: none, as
/// every gate had them); the fixture tier's budget is the one the generator recorded in the header
/// — the budget that puts half the experts on the card of every card-eligible layer — plus `extra`,
/// the card bytes a drafting load reserves out of the same budget (the draft's `card_bytes`; 0 for
/// a target-only plan). A budget lever the caller set that differs from the header is refused by
/// name, never planned under.
///
/// # Errors
/// The fixture tier's file with no budget key, or a lever that names another budget.
pub fn plan_levers(split: &Split, levers: &Levers, extra: u64) -> Result<PlanLevers, GateError> {
    budget_levers(
        Tier::from_env()?,
        header_budget(split)?,
        levers.card_budget_bytes(),
        extra,
    )
}

/// [`plan_levers`] of its three inputs: the tier, the header's budget and the caller's lever.
///
/// # Errors
/// As [`plan_levers`].
pub fn budget_levers(
    tier: Tier,
    header: Option<u64>,
    lever: Option<u64>,
    extra: u64,
) -> Result<PlanLevers, GateError> {
    match tier {
        Tier::Real => Ok(PlanLevers {
            card_budget_bytes: lever,
        }),
        Tier::Fixture => {
            let header =
                header.ok_or_else(|| format!("the fixture header has no {KEY_CARD_BUDGET}"))?;
            if let Some(l) = lever
                && l != header
            {
                return Err(format!(
                    "BLOOMERY_CARD_BUDGET={l} and the fixture's {KEY_CARD_BUDGET} is {header}: \
                     the fixture tier plans under the header's budget"
                )
                .into());
            }
            Ok(PlanLevers {
                card_budget_bytes: Some(header + extra),
            })
        }
    }
}

/// The card a plan is made on, and the machine's usable bytes with it. The real tier keeps what
/// the gates had (`real`: the 3090's bytes on the card the runner put in view). The fixture tier
/// resolves `a` against this process's devices (`--place a`): the largest visible card as the
/// device reports itself, its free bytes included.
///
/// # Errors
/// `a` finding no card.
#[cfg(feature = "gpu")]
pub fn card(
    real: impl FnOnce() -> Result<model::placement::workstation::CardSpec, GateError>,
) -> Result<model::placement::workstation::CardSpec, GateError> {
    use model::placement::workstation::CardSpec;
    use std::sync::OnceLock;

    static A: OnceLock<CardSpec> = OnceLock::new();
    match Tier::from_env()? {
        Tier::Real => real(),
        Tier::Fixture => {
            if let Some(c) = A.get() {
                return Ok(*c);
            }
            let specs = crate::generate::Place::A.on_host()?.card_specs()?;
            let card = *specs.first().ok_or("--place a resolved to no card")?;
            Ok(*A.get_or_init(|| card))
        }
    }
}

/// The move proof of the card a plan is made on: its bytes (total, the driver's reserve, the
/// resolved device, the free bytes, the holders) beside `RTX_3090`'s, the lookup every plan made
/// before the card was derived. The real tier requires them equal; the name is the one difference,
/// printed apart: the plan lookup opens the card in view by its own name.
#[cfg(feature = "gpu")]
pub fn witness_card(derived: &model::placement::workstation::CardSpec) -> bool {
    use model::placement::workstation::{CardSpec, RTX_3090};

    let bytes = |c: &CardSpec| {
        (
            c.total_bytes,
            c.driver_reserve_bytes,
            c.device,
            c.free_bytes,
            c.held_by,
        )
    };
    let ok = witness(
        "card bytes (total, driver reserve, device, free, holders)",
        bytes(derived),
        bytes(&RTX_3090),
    );
    println!(
        "named difference: the card's name {:?}, the old literal's {:?}",
        derived.name, RTX_3090.name
    );
    ok
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The five tags are the five kinds of clause the module doc names, once
    /// each, and every name prints as one word.
    #[test]
    fn five_tags_in_the_docs_order() {
        let names: Vec<&str> = Tag::ALL.iter().map(|t| t.name()).collect();
        assert_eq!(
            names,
            [
                "self-consistency",
                "oracle",
                "file-bound",
                "scale",
                "fixture-oracle"
            ]
        );
        assert!(names.iter().all(|n| !n.contains(' ')));
    }

    /// Unset and empty are the real tier, the two names are the two tiers,
    /// and anything else is refused with the value in the message.
    /// Mutant: `Some("")` read as a bad value, or `Fixture` for any other.
    #[test]
    fn the_variable_names_a_tier_or_is_refused() {
        assert_eq!(Tier::parse(None), Ok(Tier::Real));
        assert_eq!(Tier::parse(Some("")), Ok(Tier::Real));
        assert_eq!(Tier::parse(Some("real")), Ok(Tier::Real));
        assert_eq!(Tier::parse(Some("fixture")), Ok(Tier::Fixture));
        for bad in ["Fixture", "fixtures", "0", " real", "1"] {
            let e = Tier::parse(Some(bad)).expect_err(bad);
            assert_eq!(e, TierError::BadValue(bad.to_string()));
            assert!(e.to_string().contains(&format!("{bad:?}")), "{e}");
        }
    }

    /// The rule: the real tier runs every tag but the fixture's oracle, the
    /// fixture tier runs self-consistency and the fixture's oracle. Mutant:
    /// the real tier running `FixtureOracle`, or the fixture tier `Oracle`.
    #[test]
    fn each_tier_runs_its_tags() {
        for tag in Tag::ALL {
            assert_eq!(
                Tier::Real.runs(tag),
                tag != Tag::FixtureOracle,
                "real and {}",
                tag.name()
            );
            assert_eq!(
                Tier::Fixture.runs(tag),
                matches!(tag, Tag::SelfConsistency | Tag::FixtureOracle),
                "fixture and {}",
                tag.name()
            );
        }
    }

    /// A clause's fate in each tier, the five tags and an untagged clause:
    /// the real tier runs everything but a fixture-oracle clause, which it
    /// leaves to the fixture tier; the fixture tier runs self-consistency and
    /// fixture-oracle clauses, defers the other three with their tag, and
    /// refuses an untagged one. Mutant: any cell changed.
    #[test]
    fn a_clauses_fate_in_each_tier() {
        let table = [
            (Tag::SelfConsistency, Decision::Run, Decision::Run),
            (Tag::Oracle, Decision::Run, Decision::Deferred(Tag::Oracle)),
            (
                Tag::FileBound,
                Decision::Run,
                Decision::Deferred(Tag::FileBound),
            ),
            (Tag::Scale, Decision::Run, Decision::Deferred(Tag::Scale)),
            (Tag::FixtureOracle, Decision::FixtureOnly, Decision::Run),
        ];
        assert_eq!(table.len(), Tag::ALL.len());
        for (tag, real, fixture) in table {
            assert_eq!(Tier::Real.decide("c", Some(tag)), Ok(real), "{tag:?}");
            assert_eq!(Tier::Fixture.decide("c", Some(tag)), Ok(fixture), "{tag:?}");
        }
        assert_eq!(Tier::Real.decide("c", None), Ok(Decision::Run));
        assert_eq!(
            Tier::Fixture.decide("c", None),
            Err(TierError::Untagged("c".to_string()))
        );
    }

    /// What each decision prints: a deferral its [`DEFERRED`] line, a
    /// real-tier fixture-oracle clause its [`FIXTURE_ONLY`] line, a
    /// fixture-tier one its [`FIXTURE_ORACLE`] line, and any other clause that
    /// runs nothing. Mutant: a spelling moved, or a clause that runs printing.
    #[test]
    fn each_decision_prints_its_line() {
        let say = |tier: Tier, tag| {
            let d = tier
                .decide("g: ik head", Some(tag))
                .expect("a tagged clause");
            d.line("g: ik head", Some(tag))
        };
        assert_eq!(
            say(Tier::Real, Tag::FixtureOracle).as_deref(),
            Some("fixture-only(oracle): g: ik head")
        );
        assert_eq!(
            say(Tier::Fixture, Tag::FixtureOracle).as_deref(),
            Some("fixture-oracle: g: ik head")
        );
        assert_eq!(
            say(Tier::Fixture, Tag::Oracle).as_deref(),
            Some("deferred(real) oracle: g: ik head")
        );
        assert_eq!(say(Tier::Real, Tag::Oracle), None);
        assert_eq!(say(Tier::Fixture, Tag::SelfConsistency), None);
        assert_eq!(say(Tier::Real, Tag::SelfConsistency), None);
        assert!(fixture_only_line("x").starts_with(FIXTURE_ONLY));
        assert!(fixture_oracle_line("x").starts_with(FIXTURE_ORACLE));
    }

    /// A clause's call prints its decision's line to the gate's output, counts it in the tally's
    /// kind and answers whether it runs: a fixture-oracle clause in the real tier prints
    /// `fixture-only(oracle)`, is counted as left to the fixture tier and does not run; in the
    /// fixture tier it prints `fixture-oracle`, runs and is counted as run; an oracle clause
    /// in the fixture tier is deferred; a self-consistency clause prints nothing in either.
    /// Mutant: the line not written, the decision not counted, or counted in another kind.
    #[test]
    fn a_clause_call_prints_counts_and_answers() {
        let call = |tier: Tier, tag: Tag| {
            let (tally, mut out) = (Tally::new(), Vec::new());
            let runs =
                run_clause_in(tier, "g: ik head", tag, &tally, &mut out).expect("a tagged clause");
            (runs, String::from_utf8(out).expect("UTF-8"), tally.counts())
        };
        assert_eq!(
            call(Tier::Real, Tag::FixtureOracle),
            (
                false,
                "fixture-only(oracle): g: ik head\n".to_string(),
                (0, 0, 1)
            )
        );
        assert_eq!(
            call(Tier::Fixture, Tag::FixtureOracle),
            (true, "fixture-oracle: g: ik head\n".to_string(), (1, 0, 0))
        );
        assert_eq!(
            call(Tier::Fixture, Tag::Oracle),
            (
                false,
                "deferred(real) oracle: g: ik head\n".to_string(),
                (0, 1, 0)
            )
        );
        assert_eq!(
            call(Tier::Real, Tag::Oracle),
            (true, String::new(), (1, 0, 0))
        );
        assert_eq!(
            call(Tier::Fixture, Tag::SelfConsistency),
            (true, String::new(), (1, 0, 0))
        );
    }

    /// The tally counts each kind of decision once, and its line has the
    /// third term only when a clause was left to the fixture tier, so every
    /// line a gate printed before a fixture-oracle clause existed is the same
    /// bytes. Mutant: the third term printed at 0, or a decision counted in
    /// another kind.
    #[test]
    fn the_tally_counts_and_prints_three_kinds() {
        let t = Tally::new();
        for d in [
            Decision::Run,
            Decision::Run,
            Decision::Run,
            Decision::Deferred(Tag::Oracle),
            Decision::Deferred(Tag::Scale),
        ] {
            t.record(d);
        }
        assert_eq!(t.counts(), (3, 2, 0));
        t.record(Decision::FixtureOnly);
        t.record(Decision::FixtureOnly);
        assert_eq!(t.counts(), (3, 2, 2));
        assert_eq!(
            tally_text(3, 2, 0),
            "clauses: 3 ran, 2 left to the real tier"
        );
        assert_eq!(
            tally_text(0, 0, 0),
            "clauses: 0 ran, 0 left to the real tier"
        );
        assert_eq!(
            tally_text(3, 2, 2),
            "clauses: 3 ran, 2 left to the real tier, 2 left to the fixture tier"
        );
    }

    /// A clause with no tag in the fixture tier is an error that names the
    /// clause and the four tags, not a run and not a deferral.
    /// Mutant: the `(Tier::Fixture, None)` arm returns `Ok(Decision::Run)`.
    #[test]
    fn an_untagged_clause_in_the_fixture_tier_is_a_named_error() {
        let e = Tier::Fixture
            .decide("e: free run", None)
            .expect_err("an untagged clause ran in the fixture tier");
        assert_eq!(e, TierError::Untagged("e: free run".to_string()));
        let text = e.to_string();
        assert!(text.contains("\"e: free run\""), "{text}");
        for tag in Tag::ALL {
            assert!(text.contains(tag.name()), "{text}");
        }
        assert!(Tier::Fixture.clause("e: free run", None).is_err());
    }

    /// The deferred line is the prefix the batch runner counts, then the
    /// tag, then the clause, and `tools/gate-batch.sh` counts that prefix.
    /// Mutant: either side's spelling moves.
    #[test]
    fn the_deferred_line_is_what_the_batch_counts() {
        assert_eq!(
            deferred_line("h: ik argmax", Tag::Oracle),
            "deferred(real) oracle: h: ik argmax"
        );
        assert!(deferred_line("x", Tag::Scale).starts_with(&format!("{DEFERRED} ")));
        let batch = include_str!("../../../tools/gate-batch.sh");
        assert!(
            batch.contains(&format!(
                "^{}",
                DEFERRED.replace('(', "\\(").replace(')', "\\)")
            )),
            "tools/gate-batch.sh does not count the {DEFERRED} prefix"
        );
    }

    /// The fixture tier wants the generator's version key and refuses a
    /// subset file; the real tier reads nothing.
    /// Mutant: the version test dropped, or the subset test dropped.
    #[test]
    fn the_fixture_tier_opens_only_a_whole_fixture() {
        let p = Path::new("/m/f-00001-of-00001.gguf");
        let whole = ["general.name", VERSION_KEY, "bloomery.fixture.seed"];
        assert_eq!(Tier::Fixture.check_file(p, whole), Ok(()));
        assert_eq!(
            Tier::Fixture.check_file(p, ["general.name", "x.block_count"]),
            Err(TierError::NotFixture(p.display().to_string()))
        );
        let subset = [VERSION_KEY, SUBSET_KEY];
        assert_eq!(
            Tier::Fixture.check_file(p, subset),
            Err(TierError::SubsetFile(p.display().to_string()))
        );
        assert_eq!(
            Tier::Fixture.check_file(p, []),
            Err(TierError::NotFixture(p.display().to_string()))
        );
        assert_eq!(Tier::Real.check_file(p, []), Ok(()));
        assert_eq!(Tier::Real.check_file(p, subset), Ok(()));
        let e = Tier::Fixture.check_file(p, []).expect_err("not a fixture");
        assert!(e.to_string().contains("never opens a real one"), "{e}");
    }

    /// The generator names the two header keys this module reads: some file
    /// of the model crate holds each as a string literal, so a rename there
    /// is red here.
    #[test]
    fn the_generator_writes_the_keys_this_module_reads() {
        fn sources(dir: &Path, out: &mut String) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    sources(&path, out);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    out.push_str(&std::fs::read_to_string(&path).unwrap_or_default());
                }
            }
        }
        let mut text = String::new();
        sources(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../model/src"),
            &mut text,
        );
        assert!(
            !text.is_empty(),
            "crates/model/src is not where this test reads it"
        );
        for key in [VERSION_KEY, SUBSET_KEY] {
            assert!(
                text.contains(&format!("\"{key}\"")),
                "the generator in crates/model/src has no {key:?} literal"
            );
        }
    }

    /// A gate acts on the card budget in the fixture tier whatever its real list holds, once; the
    /// real tier acts on its own list alone. Mutant: the fixture arm dropped or pushing twice.
    #[test]
    fn the_fixture_tier_acts_on_the_card_budget() {
        assert_eq!(acts_in(Tier::Real, &["A"]), ["A"]);
        assert_eq!(acts_in(Tier::Fixture, &["A"]), ["A", CARD_BUDGET]);
        assert_eq!(
            acts_in(Tier::Fixture, &["A", CARD_BUDGET]),
            ["A", CARD_BUDGET]
        );
    }

    /// The real tier plans under the caller's lever whatever the header holds; the fixture tier
    /// under the header's budget plus a drafting load's card bytes, and refuses a lever that
    /// names another budget and a header with none. Mutant: the extra dropped or doubled, the
    /// lever preferred, the real tier reading the header.
    #[test]
    fn the_budget_is_the_levers_in_the_real_tier_and_the_headers_plus_the_draft_in_the_fixture() {
        let b = |t, h, l, x| budget_levers(t, h, l, x).map(|p| p.card_budget_bytes);
        assert_eq!(b(Tier::Real, Some(9), None, 5).expect("real"), None);
        assert_eq!(b(Tier::Real, None, Some(7), 5).expect("real"), Some(7));
        assert_eq!(b(Tier::Fixture, Some(9), None, 0).expect("plain"), Some(9));
        assert_eq!(
            b(Tier::Fixture, Some(9), Some(9), 5).expect("draft"),
            Some(14)
        );
        assert!(b(Tier::Fixture, Some(9), Some(8), 0).is_err());
        assert!(b(Tier::Fixture, None, None, 0).is_err());
    }

    /// A temp directory of this process for one test, empty.
    fn temp(what: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("bloomery-gates-tier-{what}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        dir
    }

    /// A GGUF file at `path` whose header holds the fixture generator's version key and `seed`
    /// (`gguf::write`'s `Writer`, no tensor).
    fn write_fixture(path: &Path, seed: u64) {
        use gguf::Value;
        use gguf::write::{Layout, Writer};
        let kvs = vec![
            (VERSION_KEY.to_string(), Value::U32(1)),
            ("bloomery.fixture.seed".to_string(), Value::U64(seed)),
        ];
        let layout =
            Layout::new(&kvs, Vec::new()).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let file =
            std::fs::File::create(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        Writer::new(file, layout)
            .and_then(Writer::finish)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    }

    /// A node-dump set at `dir` whose manifest states `model` and `fixture` (the text after the
    /// tab of its `# fixture` line) and `build`, with one tensor row and, when `complete`, the
    /// trailer.
    fn write_set(dir: &Path, model: &str, fixture: &str, build: &str, complete: bool) {
        std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        let mut lines = vec![
            format!("# model\t{model}"),
            format!("# fixture\t{fixture}"),
            format!("# build\t{build}"),
            "# arch\ttesta".to_string(),
            "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top".to_string(),
            "tensor\tl_out-0\t0\tf32\t1\t1\t1\t1\t4\t0\tADD".to_string(),
        ];
        if complete {
            lines.push("# complete\t1\t0".to_string());
        }
        let path = dir.join("MANIFEST.tsv");
        std::fs::write(&path, lines.join("\n") + "\n")
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    }

    /// The fixture file `FIXTURE_FILE` names, as a family's `runs` reads it: a static, since a
    /// `Family` holds a `fn`, and no environment variable is set.
    static FIXTURE_FILE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    fn runs_the_temp_fixture() -> Result<String, RefError> {
        FIXTURE_FILE
            .get()
            .cloned()
            .ok_or_else(|| RefError::Missing {
                path: PathBuf::new(),
                what: "the test's fixture file is not written".to_string(),
            })
    }

    fn runs_no_fixture() -> Result<String, RefError> {
        Err(RefError::Missing {
            path: PathBuf::from("/fixtures/testa"),
            what: "fixture: /fixtures/testa holds 0 first shards".to_string(),
        })
    }

    const RECIPE: &str = "BLOOMERY_TIER=fixture just dump-test [VARIANT]";

    static TEST_FAMILY: Family = Family {
        name: "fx-test",
        sets: &["fx_set", "fx_other"],
        resolve: None,
        recipe: RECIPE,
        identity: Identity::FixtureManifest,
        arch: Some("testa"),
        build: Some(refset::family::Build::Is("abc12345")),
        runs: Some(runs_the_temp_fixture),
        draft_runs: None,
        consumers: &[],
    };

    /// [`TEST_FAMILY`] with a real family's identity, whose check cannot tell a generation of the
    /// fixture.
    static REAL_FAMILY: Family = Family {
        name: "ik-test",
        sets: &["fx_set"],
        resolve: None,
        recipe: RECIPE,
        identity: Identity::Manifest,
        arch: Some("testa"),
        build: Some(refset::family::Build::Is("abc12345")),
        runs: Some(runs_the_temp_fixture),
        draft_runs: None,
        consumers: &[],
    };

    /// [`TEST_FAMILY`] whose fixture file is not there.
    static NO_FIXTURE_FAMILY: Family = Family {
        name: "fx-test",
        sets: &["fx_set", "fx_other"],
        resolve: None,
        recipe: RECIPE,
        identity: Identity::FixtureManifest,
        arch: Some("testa"),
        build: Some(refset::family::Build::Is("abc12345")),
        runs: Some(runs_no_fixture),
        draft_runs: None,
        consumers: &[],
    };

    /// A fixture-oracle clause's set, opened through the fixture family's check: a set dumped on
    /// the fixture the tree runs opens and returns its path; a set that is not there is `Missing`
    /// naming the path and the family's own recipe; one dumped on another generation of the
    /// fixture (`# fixture` differs) is `Stale`, one by another ik build `Foreign`, one with no
    /// trailer `Unfinished`, one with a row that does not parse `Malformed`, each naming the
    /// clause and its kind; and none of them opens as a deferral.
    /// Mutant: `Missing` read as `Ok`, the stale check dropped, a kind's label moved.
    #[test]
    fn a_fixture_set_opens_or_fails_by_name() {
        let root = temp("set");
        let first = root.join("f-00001-of-00001.gguf");
        write_fixture(&first, 1);
        let first_text = first.display().to_string();
        FIXTURE_FILE
            .set(first_text.clone())
            .expect("this test alone sets the fixture file");
        let ours = refset::fixture::line(&first).expect("the fixture file's line");
        let ours = ours
            .strip_prefix(refset::fixture::LINE_PREFIX)
            .expect("a fixture line");
        let theirs = {
            let other = root.join("g-00001-of-00001.gguf");
            write_fixture(&other, 2);
            let l = refset::fixture::line(&other).expect("the other fixture's line");
            l.strip_prefix(refset::fixture::LINE_PREFIX)
                .expect("a fixture line")
                .to_string()
        };
        assert_ne!(ours, theirs);
        let data = root.join("data");
        let open = |set: &str| fixture_set_in(&data, "c: ik head", &TEST_FAMILY, set);

        let e = open("fx_set").expect_err("no set in place");
        assert!(matches!(e, FixtureSetError::Missing { .. }), "{e:?}");
        assert_eq!(
            e.to_string(),
            format!(
                "fixture oracle c: ik head: set {} missing; dump it with `{RECIPE}`",
                data.join("fx_set").display()
            )
        );

        write_set(&data.join("fx_set"), &first_text, ours, "abc12345", true);
        assert_eq!(
            open("fx_set").expect("a set of this fixture"),
            data.join("fx_set")
        );

        write_set(&data.join("fx_set"), &first_text, &theirs, "abc12345", true);
        let e = open("fx_set").expect_err("another generation of the fixture");
        assert!(matches!(e.cause(), Some(RefError::Stale { .. })), "{e:?}");
        let text = e.to_string();
        assert!(
            text.starts_with("fixture oracle c: ik head: stale set: ")
                && text.contains(&format!(
                    "dumped from {}",
                    refset::fixture::LINE_PREFIX.to_owned() + &theirs
                ))
                && text.contains(&format!(
                    "the tree runs {}",
                    refset::fixture::LINE_PREFIX.to_owned() + ours
                ))
                && text.ends_with(&format!("dump it again with `{RECIPE}`")),
            "{text}"
        );

        write_set(&data.join("fx_set"), &first_text, ours, "49ef19d0", true);
        let e = open("fx_set").expect_err("another ik build");
        assert!(matches!(e.cause(), Some(RefError::Foreign { .. })), "{e:?}");
        assert!(e.to_string().contains("foreign set: "), "{e}");

        write_set(&data.join("fx_set"), &first_text, ours, "abc12345", false);
        let e = open("fx_set").expect_err("a set with no trailer");
        assert!(
            matches!(e.cause(), Some(RefError::Unfinished { .. })),
            "{e:?}"
        );
        assert!(e.to_string().contains("unfinished set: "), "{e}");

        write_set(&data.join("fx_set"), &first_text, ours, "abc12345", true);
        let manifest = data.join("fx_set").join("MANIFEST.tsv");
        let text = std::fs::read_to_string(&manifest).expect("the manifest");
        std::fs::write(&manifest, text.replace("ADD", "ADD\textra")).expect("the manifest");
        let e = open("fx_set").expect_err("a row of another width");
        assert!(
            matches!(e.cause(), Some(RefError::Malformed { .. })),
            "{e:?}"
        );
        assert!(e.to_string().contains("malformed set: "), "{e}");

        std::fs::remove_file(&manifest).expect("the manifest");
        let e = open("fx_set").expect_err("a set directory with no manifest");
        assert!(matches!(e, FixtureSetError::Missing { .. }), "{e:?}");

        std::fs::remove_dir_all(&root).expect("the temp directory");
    }

    /// The refusals that need no set on disk: a real family would open any generation of the
    /// fixture, a set of another family is not this one's, and a fixture file that is not there
    /// is named as the fixture, not as the set (the set is in place here).
    /// Mutant: the identity check dropped, or `Missing` of the fixture read as the set's.
    #[test]
    fn a_clause_names_the_family_and_the_fixture_it_needs() {
        let data = temp("family");
        write_set(
            &data.join("fx_set"),
            "/fixtures/testa/f-00001-of-00001.gguf",
            "k=v",
            "abc12345",
            true,
        );

        let e = fixture_set_in(&data, "c", &REAL_FAMILY, "fx_set").expect_err("a real family");
        assert!(
            matches!(e, FixtureSetError::NotFixtureFamily { .. }),
            "{e:?}"
        );
        assert!(e.to_string().contains(REAL_FAMILY.name), "{e}");

        let e = fixture_set_in(&data, "c", &TEST_FAMILY, "fx_nope").expect_err("another set");
        assert!(matches!(e, FixtureSetError::NotInFamily { .. }), "{e:?}");
        assert!(
            e.to_string().contains("fx_nope") && e.to_string().contains("fx-test"),
            "{e}"
        );

        let e = fixture_set_in(&data, "c: ik head", &NO_FIXTURE_FAMILY, "fx_set")
            .expect_err("no fixture file");
        assert!(matches!(e.cause(), Some(RefError::Missing { .. })), "{e:?}");
        let text = e.to_string();
        assert!(
            text.starts_with("fixture oracle c: ik head: the fixture file is missing: ")
                && text.contains("/fixtures/testa")
                && !text.contains(" missing; dump it with"),
            "{text}"
        );

        std::fs::remove_dir_all(&data).expect("the temp directory");
    }
}
