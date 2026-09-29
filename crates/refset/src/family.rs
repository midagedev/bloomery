//! A family of reference sets: one kind of set the gates read, where its sets
//! live, the recipe that writes them, which line of a set states the model
//! file it was dumped from, the ik build it must name, and the gates that
//! read it. A family's readers check a set against its row before any
//! comparison: a set of another file is [`RefError::Stale`], naming the set,
//! the file it states and the file the tree runs. Each architecture's
//! families and the table of all of them are [`crate::arch`]'s.
//!
//! The identity every writer emits is the full path of the model's first
//! shard, as the dumper was given it. The two V4.1 files name their shards
//! alike, so a basename cannot tell them apart; their directories differ.

use crate::RefError;
use std::path::{Path, PathBuf};

/// Where a set states what it was dumped from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Identity {
    /// `# model\t<path>` in the set's `MANIFEST.tsv` (`dump_ref`).
    Manifest,
    /// `# model\t<path>` and `# draft_model\t<path>` (`dump_draft`).
    ManifestAndDraft,
    /// `# model\t<path>` in an MTP draft set's `MANIFEST.tsv` (`dump_mtp`):
    /// the target file, which carries the NextN block; with
    /// `# draft_model\t<path>` too for a family whose draft is a file of its
    /// own (`Family::draft_runs`).
    MtpManifest,
    /// `model=<path>` in the `# argmax_ref` line heading the tsv (`argmax_ref`).
    ArgmaxHeader,
    /// `  model <path>` in the run's log, `<tag>.log` beside `<tag>.kld`
    /// (`tools/ref/ik-ppl.sh`).
    RunLog,
    /// `# checkpoint\t<repo>@<revision>\t<dir>`: the checkpoint's revision,
    /// which the family pins.
    Checkpoint { revision: &'static str },
}

/// The ik build a family's sets must name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Build {
    /// The pin itself.
    Is(&'static str),
    /// The pin, `+` and a patch: a tree patched on top of the pin and named
    /// after it (the draft tree's `db517b69+dsv41-draft <patch>`).
    Patched(&'static str),
}

impl Build {
    /// Whether `build`, a set's `# build` or its run's head, is this one.
    #[must_use]
    pub fn names(self, build: &str) -> bool {
        match self {
            Build::Is(pin) => build == pin,
            Build::Patched(pin) => build
                .strip_prefix(pin)
                .and_then(|p| p.strip_prefix('+'))
                .is_some_and(|patch| !patch.is_empty()),
        }
    }

    /// What a set must name, as a refusal prints it.
    #[must_use]
    pub fn want(self) -> String {
        match self {
            Build::Is(pin) => pin.to_string(),
            Build::Patched(pin) => format!("{pin}+<patch>"),
        }
    }
}

/// One family of reference sets.
#[derive(Debug)]
pub struct Family {
    /// The family's name, as `refset-check` prints it.
    pub name: &'static str,
    /// The sets in place, under [`crate::data_dir`]; a directory for a
    /// manifest family, the file (greedy) or the run's tag path (KLD)
    /// otherwise.
    pub sets: &'static [&'static str],
    /// The name a set takes for the file the tree runs, when that differs
    /// from its base name (the V4.1 node dumps' suffix, [`gguf::v41::set`]).
    pub resolve: Option<fn(&str) -> String>,
    /// The recipe that writes a set of the family.
    pub recipe: &'static str,
    /// Where a set states what it was dumped from.
    pub identity: Identity,
    /// The `# arch` every set must carry, for a manifest family.
    pub arch: Option<&'static str>,
    /// The ik build every set must name.
    pub build: Option<Build>,
    /// The model file the tree runs, for a family whose identity is a file.
    pub runs: Option<fn() -> String>,
    /// The draft file the tree runs, for [`Identity::ManifestAndDraft`] and
    /// an [`Identity::MtpManifest`] family whose draft is its own file.
    pub draft_runs: Option<fn() -> Result<String, RefError>>,
    /// The gate recipes that read the family.
    pub consumers: &'static [&'static str],
}

impl Family {
    /// The path of set `name` of this family under the data directory.
    #[must_use]
    pub fn path(&self, name: &str) -> PathBuf {
        let name = self.resolve.map_or_else(|| name.to_string(), |f| f(name));
        crate::data_dir().join(name)
    }

    /// The paths of the sets in place.
    #[must_use]
    pub fn in_place(&self) -> Vec<PathBuf> {
        self.sets.iter().map(|s| self.path(s)).collect()
    }

    /// The model file the tree runs, for a family whose identity is a file.
    pub fn runs(&self) -> Result<String, RefError> {
        self.runs.map(|f| f()).ok_or_else(|| RefError::Missing {
            path: PathBuf::new(),
            what: format!("the {} family names no model file", self.name),
        })
    }

    /// `dumped_from`, the model file set `set` states in its `line`, against
    /// the file the tree runs: [`RefError::Stale`] unless the two paths are
    /// the same string. A set that states none is stale too — nothing says
    /// which file it came from.
    pub fn check_file(
        &self,
        set: &Path,
        line: &str,
        dumped_from: Option<&str>,
    ) -> Result<(), RefError> {
        check_same(set, line, dumped_from, &self.runs()?)
    }

    /// The draft model `set` states, against the draft file the tree runs.
    pub fn check_draft(&self, set: &Path, dumped_from: Option<&str>) -> Result<(), RefError> {
        let runs = self.draft_runs.ok_or_else(|| RefError::Missing {
            path: set.to_path_buf(),
            what: format!("the {} family names no draft file", self.name),
        })?()?;
        check_same(set, "# draft_model", dumped_from, &runs)
    }

    /// `build`, the ik build set `set` names, against the family's
    /// ([`Build::names`]): [`RefError::Foreign`] otherwise. A family without
    /// a pin takes any build.
    pub fn check_build(&self, set: &Path, build: Option<&str>) -> Result<(), RefError> {
        let Some(pin) = self.build else {
            return Ok(());
        };
        if build.is_some_and(|b| pin.names(b)) {
            return Ok(());
        }
        Err(RefError::Foreign {
            set: set.display().to_string(),
            family: self.name,
            field: "build",
            got: build.unwrap_or("(no build line)").to_string(),
            want: pin.want(),
        })
    }

    /// `stated`, the checkpoint set `set` names (`<repo>@<revision>`),
    /// against the revision an [`Identity::Checkpoint`] family pins:
    /// [`RefError::Stale`] unless the revision after `@` is that one.
    pub fn check_revision(&self, set: &Path, stated: Option<&str>) -> Result<(), RefError> {
        let Identity::Checkpoint { revision } = self.identity else {
            return Err(RefError::Missing {
                path: set.to_path_buf(),
                what: format!("the {} family pins no checkpoint", self.name),
            });
        };
        let named = stated.and_then(|s| s.rsplit_once('@')).map(|(_, r)| r);
        if named == Some(revision) {
            return Ok(());
        }
        Err(RefError::Stale {
            set: set.display().to_string(),
            dumped_from: stated.map_or_else(
                || "an unstated checkpoint (the set has no # checkpoint line)".to_string(),
                str::to_string,
            ),
            runs: revision.to_string(),
        })
    }

    /// `arch`, the architecture set `set` names, against the family's: a set
    /// of a manifest family must name it.
    pub fn check_arch(&self, set: &Path, arch: Option<&str>) -> Result<(), RefError> {
        match self.arch {
            Some(want) if arch != Some(want) => Err(RefError::Foreign {
                set: set.display().to_string(),
                family: self.name,
                field: "arch",
                got: arch.unwrap_or("(no # arch line)").to_string(),
                want: want.to_string(),
            }),
            _ => Ok(()),
        }
    }
}

/// What a set states it was dumped from, once its family's check passed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    /// The model file (or, for a checkpoint family, the checkpoint) it states.
    pub dumped_from: String,
    /// The draft file, for [`Identity::ManifestAndDraft`] and an MTP family
    /// with one.
    pub draft: Option<String>,
    /// The ik build it names, where the family's sets record one.
    pub build: Option<String>,
}

impl Family {
    /// The set at `path` checked against this family through the family's
    /// reader: a directory for a manifest family, the tsv for
    /// [`Identity::ArgmaxHeader`], and for [`Identity::RunLog`] the base's
    /// `.kld` file or its tag's path. The error is the reader's or the check's.
    pub fn check_set(&self, path: &Path) -> Result<Provenance, RefError> {
        let stated = |s: Option<&str>| s.unwrap_or("-").to_string();
        match self.identity {
            Identity::Manifest => {
                let man = crate::ik::RefManifest::open(path, self)?;
                Ok(Provenance {
                    dumped_from: stated(man.header.model()),
                    draft: None,
                    build: man.build,
                })
            }
            Identity::ManifestAndDraft => {
                let set = crate::dsref::Dsref::open(path, self)?;
                Ok(Provenance {
                    dumped_from: stated(set.model.as_deref()),
                    draft: set.draft_model,
                    build: set.build,
                })
            }
            Identity::MtpManifest => {
                let set = crate::mtpref::MtpSet::open(path, self)?;
                Ok(Provenance {
                    dumped_from: stated(set.model.as_deref()),
                    draft: set.draft_model,
                    build: set.build,
                })
            }
            Identity::ArgmaxHeader => {
                crate::greedy::check(path, self)?;
                Ok(Provenance {
                    dumped_from: stated(crate::greedy::dumped_from(path)?.as_deref()),
                    draft: None,
                    build: None,
                })
            }
            Identity::RunLog => {
                let tag_path = if path.extension().is_some_and(|e| e == "kld") {
                    path.with_extension("")
                } else {
                    path.to_path_buf()
                };
                let tag = tag_path
                    .file_name()
                    .and_then(|t| t.to_str())
                    .ok_or_else(|| RefError::Missing {
                        path: path.to_path_buf(),
                        what: format!("{} names no run", path.display()),
                    })?;
                let kld = tag_path.with_file_name(format!("{tag}.kld"));
                let run = crate::kld::check(&kld, tag, self)?;
                Ok(Provenance {
                    dumped_from: stated(run.model.as_deref()),
                    draft: None,
                    build: run.head,
                })
            }
            Identity::Checkpoint { .. } => {
                let set = crate::vision::VisionSet::open(path, self)?;
                Ok(Provenance {
                    dumped_from: stated(set.checkpoint.as_deref()),
                    draft: None,
                    build: None,
                })
            }
        }
    }
}

/// [`RefError::Stale`] unless `dumped_from` is `runs`.
fn check_same(
    set: &Path,
    line: &str,
    dumped_from: Option<&str>,
    runs: &str,
) -> Result<(), RefError> {
    if dumped_from == Some(runs) {
        return Ok(());
    }
    Err(RefError::Stale {
        set: set.display().to_string(),
        dumped_from: dumped_from.map_or_else(
            || format!("an unstated file (the set has no {line} line)"),
            str::to_string,
        ),
        runs: runs.to_string(),
    })
}
