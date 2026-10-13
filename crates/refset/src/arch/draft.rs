//! The MTP draft file a family keeps beside its target: the one table of
//! such files ([`beside_drafts`]) and the rules that pick the file a run
//! opens ([`BesideDraft::pick`], [`BesideDraft::pick_for`]).
//!
//! A family whose draft is a file of its own, not a part of the target (GLM's
//! NextN block is inside the target's file, so it has no row), is one
//! [`BesideDraft`] row keyed by the architecture its target files declare. The
//! row says how the draft's file is found ([`DraftFile`]): one file for every
//! target of the architecture, or one per checkpoint the target can be made
//! from (MiMo's MTP head is trained against its own checkpoint's hidden state,
//! so MOPD and RL each have theirs). The `--hf` resolve fetches the row's
//! files ([`BesideDraft::candidates`]) beside the set's first shard, where the
//! engine's draft rule ([`BesideDraft::pick`]) opens it from. A family adds its
//! row here, by data:
//!
//! 1. a `pub static` [`BesideDraft`] with the draft's file ([`DraftFile`]; the
//!    file name of a path is what the repo's file is looked up by and what
//!    lands beside the target), and
//! 2. its entry in [`BESIDE_DRAFTS`].
//!
//! The usability rule (whether a fetched set can run the draft) is not data:
//! it reads the family's plan, which sits above this crate, so it lives in the
//! model crate (`model::arch::draft::draft_usable`, a table by architecture),
//! which refuses a row it has no rule for by name; the fetch (`gpu-gates`'
//! `model_file`) fetches no draft for it.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use super::mimo2;
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
    /// How the draft's file is found for a target.
    pub file: DraftFile,
}

/// How a family's draft file is found for a target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftFile {
    /// One file serves every target of the architecture. Its path is the
    /// family's, and also the file's identity: the file name of the path is
    /// the name the draft goes by in a repo and beside a target.
    Fixed(&'static str),
    /// One file per checkpoint the target can be made from. The target's
    /// `general.name` picks the entry; the entries' checkpoints are distinct.
    ByCheckpoint(&'static [CheckpointFile]),
}

/// The draft file of one checkpoint ([`DraftFile::ByCheckpoint`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointFile {
    /// The checkpoint's `general.name`, in the target's header and in the
    /// draft's.
    pub checkpoint: &'static str,
    /// The family's path of this checkpoint's draft, and its identity as a
    /// [`DraftFile::Fixed`] path is.
    pub path: &'static str,
}

/// A file a row's draft goes by: the path, and the checkpoint it follows
/// (`None`: the file serves every checkpoint).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub checkpoint: Option<&'static str>,
    pub path: &'static str,
}

impl Candidate {
    /// The file name the draft goes by, in a repo's listing and beside a
    /// target.
    ///
    /// # Panics
    /// When [`Candidate::path`] names no file: a row is a constant, and the
    /// table's test holds every row to one.
    #[must_use]
    pub fn name(&self) -> &'static str {
        Path::new(self.path)
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or_else(|| panic!("the draft path {:?} names no file", self.path))
    }
}

/// A target made from a checkpoint the row has no draft file for.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "no {arch} draft file is made from checkpoint {checkpoint:?}: the table knows {}",
    .known.join(", ")
)]
pub struct UnknownCheckpoint {
    pub arch: &'static str,
    pub checkpoint: String,
    pub known: Vec<&'static str>,
}

impl BesideDraft {
    /// Every file the draft goes by, in preference order: the one file of a
    /// [`DraftFile::Fixed`] row, or each checkpoint's of a
    /// [`DraftFile::ByCheckpoint`] one.
    #[must_use]
    pub fn candidates(&self) -> Vec<Candidate> {
        match self.file {
            DraftFile::Fixed(path) => vec![Candidate {
                checkpoint: None,
                path,
            }],
            DraftFile::ByCheckpoint(files) => files
                .iter()
                .map(|f| Candidate {
                    checkpoint: Some(f.checkpoint),
                    path: f.path,
                })
                .collect(),
        }
    }

    /// The MTP draft file a run of `target` opens, for a row whose file does
    /// not depend on the checkpoint ([`DraftFile::Fixed`]), the first of:
    ///
    /// 1. `set`, when given (the lever, which its reading has proven a file);
    /// 2. the file of the path's name in `target`'s directory, as the path
    ///    spells it;
    /// 3. the file of that name in the directory of `target` with its
    ///    symlinks resolved, when that directory is another one: a shard
    ///    reached through a symlink, or one a user linked by hand, keeps its
    ///    draft beside the real file, not beside the link;
    /// 4. the path.
    ///
    /// The family's identity stays the path whichever the run opens: a set is
    /// checked against that path, never against this one.
    ///
    /// # Panics
    /// On a [`DraftFile::ByCheckpoint`] row, which has no one path: the
    /// caller names the target's checkpoint ([`BesideDraft::pick_for`]).
    #[must_use]
    pub fn pick(&self, set: Option<&Path>, target: &Path) -> (PathBuf, DraftFrom) {
        match self.file {
            DraftFile::Fixed(path) => pick_path(path, set, target),
            DraftFile::ByCheckpoint(_) => panic!(
                "the {} draft file follows the target's checkpoint: pick_for names it",
                self.arch
            ),
        }
    }

    /// [`BesideDraft::pick`] for a target made from `checkpoint` (its
    /// `general.name`): the file of that checkpoint's entry, never another
    /// checkpoint's file lying beside the target. A [`DraftFile::Fixed`] row
    /// serves every checkpoint, so the pick is its own. A checkpoint the row
    /// has no entry for is refused by name, with the ones it has.
    pub fn pick_for(
        &self,
        checkpoint: &str,
        set: Option<&Path>,
        target: &Path,
    ) -> Result<(PathBuf, DraftFrom), UnknownCheckpoint> {
        match self.file {
            DraftFile::Fixed(path) => Ok(pick_path(path, set, target)),
            DraftFile::ByCheckpoint(files) => files
                .iter()
                .find(|f| f.checkpoint == checkpoint)
                .map(|f| pick_path(f.path, set, target))
                .ok_or_else(|| UnknownCheckpoint {
                    arch: self.arch,
                    checkpoint: checkpoint.to_string(),
                    known: files.iter().map(|f| f.checkpoint).collect(),
                }),
        }
    }

    /// Every path [`BesideDraft::pick`] checks when the lever is unset, in
    /// that order, whether or not a file is there: the draft's name in the
    /// target's directory, that name in the resolved directory when it is a
    /// different one, then [`BesideDraft::path`] when it is not already one
    /// of those. The `load draft=off` line names this list, so a missing
    /// sidecar says where the run looked, once per path.
    #[must_use]
    pub fn places(&self, target: &Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        let name = Path::new(self.path).file_name();
        if let (Some(name), Some(dir)) = (name, target.parent()) {
            out.push(dir.join(name));
            if let Some(resolved) = target
                .canonicalize()
                .ok()
                .and_then(|p| p.parent().map(Path::to_path_buf))
            {
                if resolved.as_path() != dir {
                    out.push(resolved.join(name));
                }
            }
        }
        let family = PathBuf::from(self.path);
        if !out.iter().any(|p| p == &family) {
            out.push(family);
        }
        out
    }
}

/// [`BesideDraft::pick`]'s rule over one path.
fn pick_path(path: &'static str, set: Option<&Path>, target: &Path) -> (PathBuf, DraftFrom) {
    if let Some(p) = set {
        return (p.to_path_buf(), DraftFrom::Lever);
    }
    let name = Path::new(path).file_name();
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
        None => (PathBuf::from(path), DraftFrom::Family),
    }
}

/// The MiMo-V2.6-Flash MTP sidecar (`mtp-<checkpoint>-Q8_0.gguf`; the repos'
/// Q4_0 and BF16 files are not rows, since a resolve fetches every file of
/// the table the repo holds). The two checkpoints' sidecars differ in
/// `general.name` alone, so a target takes its own checkpoint's. Not a row of
/// [`beside_drafts`] while the mimo2 seat runs no draft: a `--hf` resolve
/// would fetch the 2.38 GB file and nothing would read it.
pub static MIMO2_BESIDE: BesideDraft = BesideDraft {
    arch: mimo2::ARCH,
    file: DraftFile::ByCheckpoint(&[
        CheckpointFile {
            checkpoint: "MiMo-V2.6-Flash-MOPD",
            path: "/models/MiMo-V2.6-Flash-MOPD/mtp-MiMo-V2.6-Flash-MOPD-Q8_0.gguf",
        },
        CheckpointFile {
            checkpoint: "MiMo-V2.6-Flash-RL",
            path: "/models/MiMo-V2.6-Flash-RL/mtp-MiMo-V2.6-Flash-RL-Q8_0.gguf",
        },
    ]),
};

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
    use super::{
        BesideDraft, Candidate, DraftFile, DraftFrom, MIMO2_BESIDE, UnknownCheckpoint,
        beside_draft, beside_drafts,
    };
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
        assert_eq!(row.file, DraftFile::Fixed(PATH));
        assert_eq!(
            row.candidates(),
            [Candidate {
                checkpoint: None,
                path: PATH
            }]
        );
        assert_eq!(row.candidates()[0].name(), NAME);

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

    /// A fixed row serves every checkpoint: naming one changes no pick.
    #[test]
    fn a_fixed_row_serves_every_checkpoint() {
        let row = beside_draft("qwen4exp").expect("qwen4exp has a row");
        let target = Path::new("/nowhere/target-00001-of-00004.gguf");
        for checkpoint in ["a", "b"] {
            assert_eq!(
                row.pick_for(checkpoint, None, target),
                Ok(row.pick(None, target))
            );
        }
    }

    /// An architecture with no row has no draft to fetch or open: every
    /// family of the table but qwen4exp, and a string no family states
    /// ([`MIMO2_BESIDE`] is not in the table; its doc says why).
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

    /// Every row names a file, and no architecture holds two rows; a row's
    /// files go by distinct names, one a checkpoint.
    #[test]
    fn every_row_names_a_file_and_an_architecture_once() {
        let rows: Vec<&BesideDraft> = beside_drafts().to_vec();
        for (i, r) in rows.iter().enumerate() {
            let files = r.candidates();
            assert!(!files.is_empty(), "{}: no draft file", r.arch);
            for (j, c) in files.iter().enumerate() {
                assert!(!c.name().is_empty(), "{}: no file name", c.path);
                assert!(
                    files[..j]
                        .iter()
                        .all(|o| o.name() != c.name() && o.checkpoint != c.checkpoint),
                    "{}: a second file named {} for {:?}",
                    r.arch,
                    c.name(),
                    c.checkpoint
                );
                assert_eq!(
                    c.checkpoint.is_some(),
                    matches!(r.file, DraftFile::ByCheckpoint(_)),
                    "{}: the checkpoint of {}",
                    r.arch,
                    c.path
                );
            }
            assert!(
                rows[..i].iter().all(|o| o.arch != r.arch),
                "{}: a second row",
                r.arch
            );
        }
    }

    /// A row of this test's own, so no test depends on a path that exists.
    const ROW_PATH: &str = "/nowhere/mtp-linked-shard.gguf";
    const ROW_NAME: &str = "mtp-linked-shard.gguf";
    const ROW: BesideDraft = BesideDraft {
        arch: "test",
        file: DraftFile::Fixed(ROW_PATH),
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
        s.file("real", ROW_NAME);
        let target = s.link(&shard);
        assert_eq!(
            ROW.pick(None, &target),
            (s.real_dir("real").join(ROW_NAME), DraftFrom::Beside)
        );
    }

    /// A draft in the link's own directory is found first: the unresolved
    /// directory is the user's word, the resolved one the fallback.
    #[cfg(unix)]
    #[test]
    fn a_draft_beside_the_link_wins_over_one_beside_the_real_shard() {
        let s = Scratch::new("link-first");
        let shard = s.file("real", "shard-00001.gguf");
        s.file("real", ROW_NAME);
        let beside_link = s.file("links", ROW_NAME);
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
            (PathBuf::from(ROW_PATH), DraftFrom::Family)
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
            (PathBuf::from(ROW_PATH), DraftFrom::Family)
        );
        let beside = s.real_dir("only").join(ROW.name());
        assert_eq!(ROW.places(&target), vec![beside, PathBuf::from(ROW.path)]);
    }

    /// The beside path is the family's path when the target already sits in
    /// that directory, so the off line names it once.
    #[test]
    fn places_names_the_family_path_once_when_the_target_sits_there() {
        let target = Path::new("/nowhere/shard.gguf");
        assert_eq!(ROW.places(target), vec![PathBuf::from(ROW.path)]);
    }

    /// A symlink is looked at in the link's directory and then beside the
    /// real shard, and the family's path is last. The off line names all three.
    #[cfg(unix)]
    #[test]
    fn places_names_the_link_the_real_shard_and_the_family_path() {
        let s = Scratch::new("places");
        let shard = s.file("real", "shard-00001.gguf");
        let target = s.link(&shard);
        assert_eq!(
            ROW.places(&target),
            vec![
                s.0.join("links").join(ROW.name()),
                s.real_dir("real").join(ROW.name()),
                PathBuf::from(ROW.path),
            ]
        );
    }

    /// The mimo2 row, written out with the literals: one Q8_0 sidecar a
    /// checkpoint, MOPD first, the name the hub's repos hold.
    #[test]
    fn the_mimo2_row_holds_a_sidecar_for_each_checkpoint() {
        let row = &MIMO2_BESIDE;
        let files: Vec<(Option<&str>, &str)> = row
            .candidates()
            .iter()
            .map(|c| (c.checkpoint, c.name()))
            .collect();
        assert_eq!(
            files,
            [
                (
                    Some("MiMo-V2.6-Flash-MOPD"),
                    "mtp-MiMo-V2.6-Flash-MOPD-Q8_0.gguf"
                ),
                (
                    Some("MiMo-V2.6-Flash-RL"),
                    "mtp-MiMo-V2.6-Flash-RL-Q8_0.gguf"
                ),
            ]
        );
        assert_eq!(
            row.candidates()[0].path,
            "/models/MiMo-V2.6-Flash-MOPD/mtp-MiMo-V2.6-Flash-MOPD-Q8_0.gguf"
        );
    }

    /// The pick resolves the target's own checkpoint's file: the other
    /// checkpoint's file lying beside the target is not taken, the lever
    /// wins, and a checkpoint the row lacks is refused naming the ones it
    /// has.
    #[test]
    fn the_pick_is_the_targets_own_checkpoints_file() {
        const MOPD: &str = "MiMo-V2.6-Flash-MOPD";
        const RL: &str = "MiMo-V2.6-Flash-RL";
        const MOPD_FILE: &str = "mtp-MiMo-V2.6-Flash-MOPD-Q8_0.gguf";
        const RL_FILE: &str = "mtp-MiMo-V2.6-Flash-RL-Q8_0.gguf";
        let row = &MIMO2_BESIDE;
        let dir =
            std::env::temp_dir().join(format!("bloomery-beside-mimo2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        let target = dir.join("MiMo-V2.6-Flash-MOPD-MXFP4-00001-of-00002.gguf");
        let touch = |name: &str| -> PathBuf {
            let p = dir.join(name);
            std::fs::write(&p, b"").unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            p
        };

        // Only the RL sidecar is beside a MOPD target: the MOPD pick falls
        // back to the MOPD family path, and the RL target takes the file.
        let rl = touch(RL_FILE);
        assert_eq!(
            row.pick_for(MOPD, None, &target),
            Ok((
                PathBuf::from("/models/MiMo-V2.6-Flash-MOPD/mtp-MiMo-V2.6-Flash-MOPD-Q8_0.gguf"),
                DraftFrom::Family
            ))
        );
        assert_eq!(row.pick_for(RL, None, &target), Ok((rl, DraftFrom::Beside)));
        let mopd = touch(MOPD_FILE);
        assert_eq!(
            row.pick_for(MOPD, None, &target),
            Ok((mopd, DraftFrom::Beside))
        );
        let lever = dir.join("elsewhere.gguf");
        assert_eq!(
            row.pick_for(MOPD, Some(&lever), &target),
            Ok((lever, DraftFrom::Lever))
        );
        assert_eq!(
            row.pick_for("MiMo-V2.6-Flash-Other", None, &target),
            Err(UnknownCheckpoint {
                arch: "mimo2",
                checkpoint: "MiMo-V2.6-Flash-Other".to_string(),
                known: vec![MOPD, RL],
            })
        );
        std::fs::remove_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    }

    /// A row with one file a checkpoint has no single path: the
    /// checkpoint-free pick is refused by name, never answered with one
    /// checkpoint's file.
    #[test]
    #[should_panic(expected = "follows the target's checkpoint")]
    fn a_checkpoint_row_has_no_checkpoint_free_pick() {
        let row = &MIMO2_BESIDE;
        let _ = row.pick(None, Path::new("/nowhere/target.gguf"));
    }
}
