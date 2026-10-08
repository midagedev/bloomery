//! Which DeepSeek-V4.1-Flash file this tree runs: the one Rust owner of the
//! path. It lives here because every crate that opens the file — the engine,
//! the engram tables, the tokenizer, the qdot and placement tests, the gate
//! binaries — depends on this crate, and this crate on none of them.
//!
//! The shell owner is the deepseek41 model profile
//! (`tools/ref/models/deepseek41.sh`, its `V41_MODEL`); `tools/box.sh`
//! exports that choice into every box command as `BLOOMERY_V41_MODEL` (and
//! its directory as `BLOOMERY_V41_DIR`), whatever profile the command picked,
//! so the two owners name one file. [`DEFAULT`] is what a process started
//! outside `box.sh` opens.
//!
//! A command that runs the deepseek41 profile also names the file it opens as
//! `BLOOMERY_REF_MODEL` (`BLOOMERY_REF_MODEL_PROFILE=deepseek41`), and the two
//! must be one file: [`resolve`] is the single rule. Under the fixture tier
//! `tools/box.sh` exports the V4.1 fixture as `BLOOMERY_REF_MODEL` and as
//! `BLOOMERY_FIXTURE_MODEL` (only once it has checked the file is a whole
//! fixture) and still the real file as `BLOOMERY_V41_MODEL`: that one pair is
//! the fixture tier's choice and the fixture is the file; any other pair that
//! differs is a named error ([`PathError`]), never a pick between them. Another
//! profile's command (Qwen3.8, GLM) names another family's file as
//! `BLOOMERY_REF_MODEL`, and V4.1 readers under it keep the real file.
//!
//! The file is the public `Q3_K_M` set ([`PUBLIC`]), the default. The oracle
//! sets dumped from it carry the name suffix [`PUBLIC_SET_SUFFIX`]; a set of
//! any other file carries none.

use std::path::PathBuf;

/// The environment variable that names the file: shard 1 of a split set.
pub const ENV: &str = "BLOOMERY_V41_MODEL";

/// The public `Q3_K_M` file's first shard.
pub const PUBLIC: &str =
    "/models/DeepSeek-V4.1-Flash-Q3_K_M/DeepSeek-V4.1-Flash-Q3_K_M-00001-of-00009.gguf";

/// The file opened when [`ENV`] is unset or empty.
pub const DEFAULT: &str = PUBLIC;

/// The oracle set name suffix of the public file's sets.
pub const PUBLIC_SET_SUFFIX: &str = "_plain";

/// The profile whose `BLOOMERY_REF_MODEL` is the V4.1 file.
pub const PROFILE: &str = "deepseek41";

/// The variable `tools/box.sh` exports the running profile's name in.
pub const PROFILE_ENV: &str = "BLOOMERY_REF_MODEL_PROFILE";

/// The variable naming the file a command's profile opens.
pub const REF_ENV: &str = "BLOOMERY_REF_MODEL";

/// The variable `tools/box.sh` sets to the fixture tier's file once it has
/// checked it is a whole fixture; unset in the real tier.
pub const FIXTURE_ENV: &str = "BLOOMERY_FIXTURE_MODEL";

/// The variable `name`, `None` when unset or empty (an empty value counts as unset, as in the
/// profile).
fn var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

/// What decides the V4.1 file: the four variables, each `None` when unset or
/// empty (an empty value counts as unset, as in the profile).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PathEnv {
    /// `BLOOMERY_V41_MODEL`.
    pub v41: Option<String>,
    /// `BLOOMERY_REF_MODEL`.
    pub reference: Option<String>,
    /// `BLOOMERY_REF_MODEL_PROFILE`.
    pub profile: Option<String>,
    /// `BLOOMERY_FIXTURE_MODEL`.
    pub fixture: Option<String>,
}

impl PathEnv {
    /// This process's variables.
    #[must_use]
    pub fn from_env() -> PathEnv {
        PathEnv {
            v41: var(ENV),
            reference: var(REF_ENV),
            profile: var(PROFILE_ENV),
            fixture: var(FIXTURE_ENV),
        }
    }

    /// Whether the command runs the deepseek41 profile: the one whose
    /// `BLOOMERY_REF_MODEL` is the V4.1 file.
    #[must_use]
    pub fn runs_v41(&self) -> bool {
        self.profile.as_deref() == Some(PROFILE)
    }
}

/// `BLOOMERY_V41_MODEL` and `BLOOMERY_REF_MODEL` name two files under the
/// deepseek41 profile, and the fixture tier did not choose the second.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathError {
    /// `BLOOMERY_V41_MODEL`, else [`DEFAULT`].
    pub v41: String,
    /// `BLOOMERY_REF_MODEL`.
    pub reference: String,
    /// `BLOOMERY_FIXTURE_MODEL`, when set.
    pub fixture: Option<String>,
}

impl std::fmt::Display for PathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{ENV}={} and {REF_ENV}={} name two files under the {PROFILE} profile, and \
             {FIXTURE_ENV}={} does not make the second the fixture tier's file: a V4.1 run \
             opens one file (gguf::v41 is its owner), so set one of them, or run the fixture \
             tier through tools/box.sh with BLOOMERY_TIER=fixture",
            self.v41,
            self.reference,
            self.fixture.as_deref().unwrap_or("unset")
        )
    }
}

impl std::error::Error for PathError {}

/// The V4.1 first shard `env` names: the single rule of the module doc.
///
/// # Errors
/// [`PathError`] when the deepseek41 profile's `BLOOMERY_REF_MODEL` is
/// another file than `BLOOMERY_V41_MODEL` and is not the file the fixture
/// tier resolved.
pub fn resolve(env: &PathEnv) -> Result<String, PathError> {
    let v41 = env.v41.clone().unwrap_or_else(|| DEFAULT.to_string());
    if !env.runs_v41() {
        return Ok(v41);
    }
    match env.reference.as_deref() {
        None => Ok(v41),
        Some(r) if r == v41 => Ok(v41),
        Some(r) if env.fixture.as_deref() == Some(r) => Ok(r.to_string()),
        Some(r) => Err(PathError {
            v41,
            reference: r.to_string(),
            fixture: env.fixture.clone(),
        }),
    }
}

/// The V4.1 first shard to open ([`resolve`] of this process's variables).
///
/// # Errors
/// [`PathError`].
pub fn try_model() -> Result<String, PathError> {
    resolve(&PathEnv::from_env())
}

/// The V4.1 first shard to open: [`try_model`], whose [`PathError`] is the
/// panic's message (a reader that cannot return one stops by name rather than
/// open a file the run did not choose).
#[must_use]
pub fn model() -> String {
    try_model().unwrap_or_else(|e| panic!("{e}"))
}

/// The directory of [`model`]: where every shard of the split set lies.
#[must_use]
pub fn dir() -> PathBuf {
    let m = PathBuf::from(model());
    m.parent().map_or_else(|| PathBuf::from("."), PathBuf::from)
}

/// The oracle set name suffix of the sets dumped from `model`: the public
/// file's [`PUBLIC_SET_SUFFIX`], nothing for any other file. A set opened for
/// another file than it was dumped from is caught by its `# model` line, not
/// by its name.
#[must_use]
pub fn set_suffix_of(model: &str) -> &'static str {
    if model == PUBLIC {
        PUBLIC_SET_SUFFIX
    } else {
        ""
    }
}

/// The name of V4.1 oracle set `base` (`ref_deepseek41`,
/// `ref_deepseek41_d1_every_node`, …) for the file [`model`] names.
#[must_use]
pub fn set(base: &str) -> String {
    format!("{base}{}", set_suffix_of(&model()))
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT, PROFILE, PUBLIC, PUBLIC_SET_SUFFIX, PathEnv, PathError, resolve, set_suffix_of,
    };

    const REAL: &str = "/models/real/real-00001-of-00009.gguf";
    const FIXTURE: &str = "/models/fixtures/v41/v41-fixture-00001-of-00004.gguf";

    fn env(
        v41: Option<&str>,
        reference: Option<&str>,
        profile: Option<&str>,
        fixture: Option<&str>,
    ) -> PathEnv {
        let own = |s: Option<&str>| s.map(str::to_string);
        PathEnv {
            v41: own(v41),
            reference: own(reference),
            profile: own(profile),
            fixture: own(fixture),
        }
    }

    /// The public file's sets carry the suffix, any other file's none. The
    /// default is the public file, so an unset variable opens the suffixed
    /// sets.
    #[test]
    fn set_suffix_follows_the_file() {
        assert_eq!(set_suffix_of(PUBLIC), PUBLIC_SET_SUFFIX);
        assert_eq!(set_suffix_of("/models/other/other-00001-of-00009.gguf"), "");
        assert_eq!(set_suffix_of(DEFAULT), PUBLIC_SET_SUFFIX);
    }

    /// Unset is the default file, and a V4.1 run whose two variables name one
    /// file opens it, with the reference unset or equal. Mutant: the default
    /// replaced by the reference.
    #[test]
    fn one_file_is_the_file() {
        assert_eq!(
            resolve(&env(None, None, None, None)).as_deref(),
            Ok(DEFAULT)
        );
        assert_eq!(
            resolve(&env(Some(REAL), None, Some(PROFILE), None)).as_deref(),
            Ok(REAL)
        );
        assert_eq!(
            resolve(&env(Some(REAL), Some(REAL), Some(PROFILE), None)).as_deref(),
            Ok(REAL)
        );
    }

    /// The two variables name two files under the deepseek41 profile: a named
    /// error carrying both, not a pick of either — with no fixture, with
    /// another file's fixture, and with the fixture variable naming the first.
    /// Mutant: `resolve` returns the reference (or the V4.1 variable) when
    /// they differ.
    #[test]
    fn two_files_are_an_error_never_a_pick() {
        for fixture in [
            None,
            Some("/models/fixtures/glm5next/g-00001-of-00002.gguf"),
            Some(REAL),
        ] {
            let e = resolve(&env(Some(REAL), Some(FIXTURE), Some(PROFILE), fixture))
                .expect_err("two files, no fixture tier choice");
            assert_eq!(
                e,
                PathError {
                    v41: REAL.to_string(),
                    reference: FIXTURE.to_string(),
                    fixture: fixture.map(str::to_string),
                }
            );
            let text = e.to_string();
            assert!(text.contains(REAL) && text.contains(FIXTURE), "{text}");
            assert!(text.contains("BLOOMERY_V41_MODEL"), "{text}");
        }
        // The default stands for an unset V4.1 variable: the reference differs from it too.
        assert!(resolve(&env(None, Some(FIXTURE), Some(PROFILE), None)).is_err());
    }

    /// The fixture tier's pair — the reference the fixture box.sh resolved,
    /// the V4.1 variable still the real file — opens the fixture. Mutant: the
    /// fixture variable ignored (the real file opens under the fixture tier,
    /// the defect this owner closes), or the reference taken without it.
    #[test]
    fn the_fixture_tiers_pair_opens_the_fixture() {
        assert_eq!(
            resolve(&env(
                Some(REAL),
                Some(FIXTURE),
                Some(PROFILE),
                Some(FIXTURE)
            ))
            .as_deref(),
            Ok(FIXTURE)
        );
        assert_eq!(
            resolve(&env(None, Some(FIXTURE), Some(PROFILE), Some(FIXTURE))).as_deref(),
            Ok(FIXTURE)
        );
    }

    /// Another profile's `BLOOMERY_REF_MODEL` is another family's file and
    /// never the V4.1 file, whatever the fixture variable says: the tokenizer,
    /// engram and qdot tests under a Qwen3.8 command keep the real V4.1 file.
    /// Mutant: the profile test dropped.
    #[test]
    fn another_profiles_files_are_not_the_v41_file() {
        let qwen_fixture = "/models/fixtures/qwen38/q-00001-of-00001.gguf";
        for profile in [None, Some("qwen4exp"), Some("glm5next")] {
            assert_eq!(
                resolve(&env(
                    Some(REAL),
                    Some(qwen_fixture),
                    profile,
                    Some(qwen_fixture)
                ))
                .as_deref(),
                Ok(REAL)
            );
        }
    }
}
