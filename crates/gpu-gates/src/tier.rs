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
//!
//! The `real` tier runs every clause, tagged or not: nothing changes for a
//! gate that declares no tags. The `fixture` tier runs the self-consistency
//! clauses and defers the other three, each deferral a [`DEFERRED`] line the
//! batch runner (`tools/gate-batch.sh`) counts; a clause with no tag in the
//! fixture tier is a named error, never a clause that quietly ran or
//! quietly did not.

use std::fmt;
use std::path::Path;

/// The variable the tier is read from.
pub const ENV: &str = "BLOOMERY_TIER";

/// A line that starts with this is a clause the fixture tier left to the real
/// tier. `tools/gate-batch.sh` counts these lines per item.
pub const DEFERRED: &str = "deferred(real)";

/// The key the generator writes into every fixture's header.
pub const VERSION_KEY: &str = "bloomery.fixture.version";

/// The key the generator writes into a file that holds only some of the
/// planned tensors; the engine must not run such a file.
pub const SUBSET_KEY: &str = "bloomery.fixture.subset";

/// Which tier a gate runs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// The files the model profiles name; every clause runs.
    Real,
    /// A generated small file; only the self-consistency clauses run.
    Fixture,
}

/// What a clause answers (the module doc names the four).
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
}

impl Tag {
    /// Every tag, in the order the module doc names them.
    pub const ALL: [Tag; 4] = [
        Tag::SelfConsistency,
        Tag::Oracle,
        Tag::FileBound,
        Tag::Scale,
    ];

    /// The tag's name as a [`DEFERRED`] line prints it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Tag::SelfConsistency => "self-consistency",
            Tag::Oracle => "oracle",
            Tag::FileBound => "file-bound",
            Tag::Scale => "scale",
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
            Tier::Real => true,
            Tier::Fixture => tag == Tag::SelfConsistency,
        }
    }

    /// What this tier does with `clause`, which declares `tag` (`None`: no
    /// tag). The real tier runs every clause. The fixture tier runs a
    /// self-consistency clause, defers the others, and refuses an untagged
    /// one.
    ///
    /// # Errors
    /// [`TierError::Untagged`] for a clause with no tag in the fixture tier.
    pub fn decide(self, clause: &str, tag: Option<Tag>) -> Result<Decision, TierError> {
        match (self, tag) {
            (Tier::Real, _) => Ok(Decision::Run),
            (Tier::Fixture, None) => Err(TierError::Untagged(clause.to_string())),
            (Tier::Fixture, Some(tag)) if self.runs(tag) => Ok(Decision::Run),
            (Tier::Fixture, Some(tag)) => Ok(Decision::Deferred(tag)),
        }
    }

    /// The gate's call for one clause: `Ok(true)` when it runs the clause,
    /// `Ok(false)` after printing the clause's [`DEFERRED`] line.
    ///
    /// # Errors
    /// As [`Tier::decide`].
    pub fn clause(self, clause: &str, tag: Option<Tag>) -> Result<bool, TierError> {
        match self.decide(clause, tag)? {
            Decision::Run => Ok(true),
            Decision::Deferred(tag) => {
                println!("{}", deferred_line(clause, tag));
                Ok(false)
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The four tags are the four kinds of clause the design names, once
    /// each, and every name prints as one word.
    #[test]
    fn four_tags_in_the_designs_order() {
        let names: Vec<&str> = Tag::ALL.iter().map(|t| t.name()).collect();
        assert_eq!(names, ["self-consistency", "oracle", "file-bound", "scale"]);
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

    /// The rule: the real tier runs every tag, the fixture tier only
    /// self-consistency. Mutant: `Oracle` run in the fixture tier.
    #[test]
    fn the_fixture_tier_runs_only_self_consistency() {
        for tag in Tag::ALL {
            assert!(Tier::Real.runs(tag), "real runs {}", tag.name());
            assert_eq!(
                Tier::Fixture.runs(tag),
                tag == Tag::SelfConsistency,
                "fixture and {}",
                tag.name()
            );
        }
    }

    /// A clause's fate: run, or deferred with its tag; the real tier runs a
    /// clause whatever it declares.
    #[test]
    fn a_tagged_clause_runs_or_is_deferred() {
        assert_eq!(
            Tier::Fixture.decide("eager = replay", Some(Tag::SelfConsistency)),
            Ok(Decision::Run)
        );
        for tag in [Tag::Oracle, Tag::FileBound, Tag::Scale] {
            assert_eq!(
                Tier::Fixture.decide("c", Some(tag)),
                Ok(Decision::Deferred(tag))
            );
        }
        for tag in Tag::ALL {
            assert_eq!(Tier::Real.decide("c", Some(tag)), Ok(Decision::Run));
        }
        assert_eq!(Tier::Real.decide("c", None), Ok(Decision::Run));
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
}
