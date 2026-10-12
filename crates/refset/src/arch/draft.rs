//! The MTP draft file a family keeps beside its target: the one table of
//! such files ([`beside_drafts`]) and the rule that picks the file a run
//! opens ([`BesideDraft::pick`]).
//!
//! A family whose draft is a file of its own, not a part of the target (GLM's
//! NextN block is inside the target's file, so it has no row), is one
//! [`BesideDraft`] row keyed by the architecture its target files declare. The
//! `--hf` resolve fetches the row's file ([`BesideDraft::name`]) beside the
//! set's first shard, where the engine's draft rule ([`BesideDraft::pick`])
//! opens it from. A family adds its row here, by data:
//!
//! 1. a `pub static` [`BesideDraft`] in the family's module, with the draft's
//!    path (its file name is what the repo's file is looked up by and what
//!    lands beside the target), and
//! 2. its entry in [`BESIDE_DRAFTS`].
//!
//! The usability rule (whether a fetched set can run the draft) is not data:
//! it reads the family's plan, which sits above this crate, so it lives with
//! the fetch (`gpu-gates`' `model_file::draft_usable`), which refuses a row it
//! has no rule for by name and fetches no draft for it.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use super::qwen4exp;

/// Where the MTP draft file a run opens came from ([`BesideDraft::pick`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftFrom {
    /// `BLOOMERY_MTP_DRAFT`.
    Lever,
    /// The file of the row's draft name in the target's directory, or in the
    /// directory the target's symlinks resolve to.
    Beside,
    /// The row's path itself: the lever unset and no file of its name beside
    /// the target.
    Family,
}

impl DraftFrom {
    /// Where the file came from, as an error about opening it says.
    #[must_use]
    pub fn describe(self) -> &'static str {
        match self {
            DraftFrom::Lever => "BLOOMERY_MTP_DRAFT",
            DraftFrom::Beside => "the shared draft's name beside the target",
            DraftFrom::Family => {
                "the family's path: BLOOMERY_MTP_DRAFT is unset and no file of the shared draft's \
                 name is beside the target"
            }
        }
    }
}

/// One family's beside-file MTP draft.
#[derive(Debug)]
pub struct BesideDraft {
    /// The `general.architecture` its target files declare.
    pub arch: &'static str,
    /// The family's path of the draft file, which is also its identity: the
    /// file name of the path is the name the draft goes by in a repo and
    /// beside a target.
    pub path: &'static str,
}

impl BesideDraft {
    /// The file name the draft goes by, in a repo's listing and beside a
    /// target.
    ///
    /// # Panics
    /// When [`BesideDraft::path`] names no file: a row is a constant, and the
    /// table's test holds every row to one.
    #[must_use]
    pub fn name(&self) -> &'static str {
        Path::new(self.path)
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or_else(|| panic!("the draft path {:?} names no file", self.path))
    }

    /// The MTP draft file a run of `target` opens, the first of:
    ///
    /// 1. `set`, when given (the lever, which its reading has proven a file);
    /// 2. the file of [`BesideDraft::name`] in `target`'s directory, as the
    ///    path spells it;
    /// 3. the file of that name in the directory of `target` with its
    ///    symlinks resolved, when that directory is another one: a shard
    ///    reached through a symlink, or one a user linked by hand, keeps its
    ///    draft beside the real file, not beside the link;
    /// 4. [`BesideDraft::path`].
    ///
    /// The family's identity stays the path whichever the run opens: a set is
    /// checked against that path, never against this one.
    #[must_use]
    pub fn pick(&self, set: Option<&Path>, target: &Path) -> (PathBuf, DraftFrom) {
        if let Some(p) = set {
            return (p.to_path_buf(), DraftFrom::Lever);
        }
        let name = Path::new(self.path).file_name();
        let named_in = |dir: &Path| name.map(|n| dir.join(n)).filter(|p| p.is_file());
        let unresolved = target.parent();
        let beside = unresolved.and_then(named_in).or_else(|| {
            let resolved = target.canonicalize().ok()?;
            resolved
                .parent()
                .filter(|dir| Some(*dir) != unresolved)
                .and_then(named_in)
        });
        match beside {
            Some(p) => (p, DraftFrom::Beside),
            None => (PathBuf::from(self.path), DraftFrom::Family),
        }
    }
}

/// Every family's beside-file draft: the rows a `--hf` resolve looks for in a
/// repo.
static BESIDE_DRAFTS: &[&BesideDraft] = &[&qwen4exp::mtp::BESIDE];

/// Every beside-file draft row.
#[must_use]
pub fn beside_drafts() -> &'static [&'static BesideDraft] {
    BESIDE_DRAFTS
}

/// The beside-file draft of the architecture `arch`; none for a family that
/// has none.
#[must_use]
pub fn beside_draft(arch: &str) -> Option<&'static BesideDraft> {
    BESIDE_DRAFTS.iter().copied().find(|d| d.arch == arch)
}

#[cfg(test)]
mod tests {
    use super::{BesideDraft, DraftFrom, beside_draft, beside_drafts};
    use crate::arch::all;
    use std::path::{Path, PathBuf};

    /// The qwen4exp row, as the function it replaced answered, written out
    /// here with the literals: the file name, the path and the three
    /// `DraftFrom` cases. A wrong name in the row moves none of these.
    #[test]
    fn the_qwen4exp_row_answers_as_the_function_it_replaced_did() {
        const PATH: &str = "/models/Qwen3.8-Flash-Next/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";
        const NAME: &str = "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";
        // The replaced `draft_file`, body for body.
        fn old(set: Option<&Path>, target: &Path) -> (PathBuf, DraftFrom) {
            if let Some(p) = set {
                return (p.to_path_buf(), DraftFrom::Lever);
            }
            let beside = Path::new(PATH)
                .file_name()
                .zip(target.parent())
                .map(|(name, dir)| dir.join(name))
                .filter(|p| p.is_file());
            match beside {
                Some(p) => (p, DraftFrom::Beside),
                None => (PathBuf::from(PATH), DraftFrom::Family),
            }
        }

        let row = beside_draft("qwen4exp").expect("qwen4exp has a row");
        assert_eq!(row.path, PATH);
        assert_eq!(row.name(), NAME);

        let dir = std::env::temp_dir().join(format!("bloomery-beside-row-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        let target = dir.join("target-00001-of-00004.gguf");
        let other = dir.join("elsewhere.gguf");
        for set in [None, Some(other.as_path())] {
            assert_eq!(row.pick(set, &target), old(set, &target), "no file beside");
        }
        assert_eq!(
            row.pick(None, &target),
            (PathBuf::from(PATH), DraftFrom::Family)
        );
        let beside = dir.join(NAME);
        std::fs::write(&beside, b"").unwrap_or_else(|e| panic!("{}: {e}", beside.display()));
        for set in [None, Some(other.as_path())] {
            assert_eq!(row.pick(set, &target), old(set, &target), "a file beside");
        }
        assert_eq!(row.pick(None, &target), (beside, DraftFrom::Beside));
        assert_eq!(row.pick(Some(&other), &target), (other, DraftFrom::Lever));
        std::fs::remove_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    }

    /// An architecture with no row has no draft to fetch or open: every
    /// family of the table but qwen4exp, and a string no family states.
    #[test]
    fn a_family_without_a_row_has_no_beside_draft() {
        for f in all() {
            let Some(arch) = f.arch else { continue };
            assert_eq!(
                beside_draft(arch).is_some(),
                arch == "qwen4exp",
                "{}: the row's presence for {arch}",
                f.name
            );
        }
        assert!(beside_draft("no-such-arch").is_none());
    }

    /// Every row names a file, and no architecture holds two rows.
    #[test]
    fn every_row_names_a_file_and_an_architecture_once() {
        let rows: Vec<&BesideDraft> = beside_drafts().to_vec();
        for (i, r) in rows.iter().enumerate() {
            assert!(!r.name().is_empty(), "{}: no file name", r.path);
            assert!(
                rows[..i].iter().all(|o| o.arch != r.arch),
                "{}: a second row",
                r.arch
            );
        }
    }

    /// A row of this test's own, so no test depends on a path that exists.
    const ROW: BesideDraft = BesideDraft {
        arch: "test",
        path: "/nowhere/mtp-linked-shard.gguf",
    };

    /// A directory of the test's own, removed when the test ends, passed or not.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(test: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("bloomery-beside-{test}-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
            Scratch(dir)
        }

        /// An empty file `name` in the subdirectory `dir`, made as needed.
        fn file(&self, dir: &str, name: &str) -> PathBuf {
            let d = self.0.join(dir);
            std::fs::create_dir_all(&d).unwrap_or_else(|e| panic!("{}: {e}", d.display()));
            let p = d.join(name);
            std::fs::write(&p, b"").unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            p
        }

        /// `links/<name>`, a symlink to `real`.
        #[cfg(unix)]
        fn link(&self, real: &Path) -> PathBuf {
            let d = self.0.join("links");
            std::fs::create_dir_all(&d).unwrap_or_else(|e| panic!("{}: {e}", d.display()));
            let p = d.join(real.file_name().expect("a shard names a file"));
            std::os::unix::fs::symlink(real, &p)
                .unwrap_or_else(|e| panic!("{} -> {}: {e}", p.display(), real.display()));
            p
        }

        /// `dir`, with its symlinks resolved, as the run opens it.
        fn real_dir(&self, dir: &str) -> PathBuf {
            let d = self.0.join(dir);
            d.canonicalize()
                .unwrap_or_else(|e| panic!("{}: {e}", d.display()))
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            // A directory left behind is litter, not a result: the test's own
            // assertions already said what they had to.
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// `-m` a symlink whose directory holds no draft: the draft beside the
    /// real shard is the run's, as the real directory spells it.
    #[cfg(unix)]
    #[test]
    fn a_symlinked_target_finds_the_draft_beside_the_real_shard() {
        let s = Scratch::new("link-real");
        let shard = s.file("real", "shard-00001.gguf");
        s.file("real", ROW.name());
        let target = s.link(&shard);
        assert_eq!(
            ROW.pick(None, &target),
            (s.real_dir("real").join(ROW.name()), DraftFrom::Beside)
        );
    }

    /// A draft in the link's own directory is found first: the unresolved
    /// directory is the user's word, the resolved one the fallback.
    #[cfg(unix)]
    #[test]
    fn a_draft_beside_the_link_wins_over_one_beside_the_real_shard() {
        let s = Scratch::new("link-first");
        let shard = s.file("real", "shard-00001.gguf");
        s.file("real", ROW.name());
        let beside_link = s.file("links", ROW.name());
        let target = s.link(&shard);
        assert_eq!(ROW.pick(None, &target), (beside_link, DraftFrom::Beside));
    }

    /// A symlink whose real directory holds no draft either is the family's
    /// path.
    #[cfg(unix)]
    #[test]
    fn a_symlink_with_no_draft_beside_either_shard_is_the_family_path() {
        let s = Scratch::new("link-none");
        let shard = s.file("real", "shard-00001.gguf");
        let target = s.link(&shard);
        assert_eq!(
            ROW.pick(None, &target),
            (PathBuf::from(ROW.path), DraftFrom::Family)
        );
    }

    /// A target that is no symlink and has no draft beside it is the family's
    /// path, as before the resolved directory was looked at.
    #[test]
    fn a_plain_target_with_no_draft_beside_is_the_family_path() {
        let s = Scratch::new("plain-none");
        let target = s.file("only", "shard-00001.gguf");
        assert_eq!(
            ROW.pick(None, &target),
            (PathBuf::from(ROW.path), DraftFrom::Family)
        );
    }
}
