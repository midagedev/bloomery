//! The oracle table, one per architecture: which reference dump sets the
//! gates read and the tap names more than one gate asks their manifests for.
//! These are values that differ per architecture, so each is a row here and
//! not a literal in each binary (docs/arch-split.md, 「오라클·탭」); the
//! reader of a set (`refset::ik`) stays shared.

pub mod deepseek2;
pub mod deepseek41;
pub mod qwen3moe;

use crate::{GateError, RefManifest, ref_dir_named};
use model::arch::Arch;
use refset::family::Family;

/// One architecture's reference sets and shared tap names. A set name is a
/// directory under the data directory ([`crate::ref_dir_named`]); a set the
/// architecture has none of is `None`, never a name.
pub struct Oracle {
    /// The architecture the sets were dumped from.
    pub arch: Arch,
    /// ik's CUDA dump with the v2 manifest columns and logical twins.
    pub cuda_set: Option<&'static str>,
    /// ik's CPU dump.
    pub cpu_set: &'static str,
    /// The pre-v2 CUDA dump: plain files only, no logical twins.
    pub legacy_cuda_set: Option<&'static str>,
    /// The decode-step sets: each one step after a prefill, the step's
    /// position in its `# decode_pos`. Opened with [`Oracle::open_named`].
    pub step_sets: &'static [&'static str],
    /// Tap names asked of a manifest by more than one gate binary. A name
    /// that only one gate reads stays in that gate.
    pub taps: &'static [&'static str],
}

/// Which of a row's sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Set {
    Cuda,
    Cpu,
    LegacyCuda,
}

impl Oracle {
    /// The name of set `which`; an architecture without that set is an error
    /// naming both.
    pub fn set_name(&self, which: Set) -> Result<&'static str, GateError> {
        let name = match which {
            Set::Cuda => self.cuda_set,
            Set::Cpu => Some(self.cpu_set),
            Set::LegacyCuda => self.legacy_cuda_set,
        };
        name.ok_or_else(|| format!("oracle: {} has no {which:?} set", self.arch.name()).into())
    }

    /// The reference-set family this architecture's node dumps belong to,
    /// when the table holds one ([`refset::arch::node_dumps`]): every set is
    /// then opened through the family's check ([`RefManifest::open`]) —
    /// complete, dumped from the file the tree runs, of the family's
    /// architecture and ik build.
    pub fn family(&self) -> Option<&'static Family> {
        refset::arch::node_dumps(self.arch.name())
    }

    /// Open set `which`: its whole manifest ([`RefManifest::read`]) from the
    /// data directory. An architecture with a [`family`](Self::family) opens
    /// it through the family's check. Otherwise a manifest whose `# arch`
    /// line names another architecture is an error naming both; one without
    /// the line (a set dumped before the dumper wrote it) is accepted.
    pub fn open(&self, which: Set) -> Result<RefManifest, GateError> {
        let dir = ref_dir_named(self.set_name(which)?);
        if let Some(family) = self.family() {
            return Ok(RefManifest::open(&dir, family)?);
        }
        let man = RefManifest::read(&dir)?;
        self.check_arch(&man)?;
        Ok(man)
    }

    /// Open the set named `set` under the data directory — a decode-step set
    /// of [`step_sets`](Self::step_sets), say — through the family's check
    /// when the architecture has one. Otherwise its `# arch` line must name
    /// this table's architecture: every set opened by name was dumped after
    /// the dumper wrote that line, so one without it is not the set named.
    pub fn open_named(&self, set: &str) -> Result<RefManifest, GateError> {
        let dir = ref_dir_named(set);
        if let Some(family) = self.family() {
            return Ok(RefManifest::open(&dir, family)?);
        }
        let man = RefManifest::read(&dir)?;
        match man.arch.as_deref() {
            Some(a) if a == self.arch.name() => Ok(man),
            a => Err(format!(
                "oracle: {} has # arch {a:?}, but the {} table names it",
                man.dir.display(),
                self.arch.name()
            )
            .into()),
        }
    }

    /// `man`'s `# arch` line is this row's architecture or absent.
    fn check_arch(&self, man: &RefManifest) -> Result<(), GateError> {
        match man.arch.as_deref() {
            Some(a) if a != self.arch.name() => Err(format!(
                "oracle: {} was dumped from a {a} model, but the {} table names it",
                man.dir.display(),
                self.arch.name()
            )
            .into()),
            _ => Ok(()),
        }
    }
}

/// The table of architecture `a`. Every architecture the engine names has a
/// row; one without reference sets would be an error naming it.
pub fn for_arch(a: Arch) -> Result<&'static Oracle, GateError> {
    match a {
        Arch::Deepseek2 => Ok(&deepseek2::ORACLE),
        Arch::Deepseek41 => Ok(&deepseek41::ORACLE),
        Arch::Qwen3moe => Ok(&qwen3moe::ORACLE),
        Arch::Qwen35moe => Err(
            "qwen35moe has no oracle row: its gate reads the sets through refset's family".into(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::for_arch;
    use crate::{GateError, RefError, RefManifest, ref_dir_named};
    use model::arch::Arch;

    /// An architecture whose node dumps have a family in the reference-set
    /// table names that family's sets: the row's CPU set and decode-step
    /// sets are the family's, in order, and each resolves to the directory
    /// the family's own reader opens.
    #[test]
    fn a_rows_sets_are_its_familys() -> Result<(), GateError> {
        let mut tied = 0usize;
        for a in [Arch::Deepseek2, Arch::Deepseek41, Arch::Qwen3moe] {
            let o = for_arch(a)?;
            let Some(f) = o.family() else {
                continue;
            };
            let row: Vec<&str> = std::iter::once(o.cpu_set)
                .chain(o.step_sets.iter().copied())
                .collect();
            assert_eq!(
                row,
                f.sets,
                "{}: the oracle row against {}",
                a.name(),
                f.name
            );
            for &set in f.sets {
                assert_eq!(ref_dir_named(set), f.path(set), "{set}");
            }
            tied += 1;
        }
        assert!(
            tied > 0,
            "no architecture of the table has a node-dump family"
        );
        Ok(())
    }

    /// A set opened through the table is of the table's architecture. On a
    /// row without a family, a manifest naming another architecture is an
    /// error naming both; one without the `# arch` line is accepted by
    /// `open` — a set dumped before the dumper wrote it — and refused by
    /// `open_named`. On the V4.1 row every set opens through its family: of
    /// another model file it is `Stale`, of another architecture `Foreign`,
    /// and without its trailer `Unfinished`.
    #[test]
    fn a_set_of_another_architecture_is_refused() -> Result<(), GateError> {
        let dir = std::env::temp_dir().join(format!("bloomery-open-named-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let write = |header: &str| -> Result<(), GateError> {
            let rows = "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\n\
                        tensor\tx\t0\tf32\t1\t1\t1\t1\t4\t0\tNONE\n";
            std::fs::write(dir.join("MANIFEST.tsv"), format!("{header}{rows}"))?;
            Ok(())
        };
        let read = |header: &str| -> Result<RefManifest, GateError> {
            write(header)?;
            Ok(RefManifest::read(&dir)?)
        };
        // An absolute set name is the directory itself.
        let set = dir.to_str().ok_or("temp dir is not UTF-8")?;

        let q = for_arch(Arch::Qwen3moe)?;
        let err = q
            .check_arch(&read("# arch\tdeepseek2\n")?)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("deepseek2") && err.contains("qwen3moe"),
            "{err}"
        );
        assert!(q.check_arch(&read("# arch\tqwen3moe\n")?).is_ok());
        assert!(q.check_arch(&read("")?).is_ok());
        write("# arch\tqwen3moe\n")?;
        assert!(q.open_named(set).is_ok());
        write("")?;
        assert!(q.open_named(set).is_err());

        let o = for_arch(Arch::Deepseek41)?;
        let runs = gguf::v41::model();
        let build = o
            .family()
            .and_then(|f| f.build)
            .ok_or("the deepseek41 row has no family with a build")?
            .want();
        let refusal = |header: String| -> Result<RefError, GateError> {
            write(&header)?;
            match o.open_named(set) {
                Ok(_) => Err(format!("{header:?} opened").into()),
                Err(e) => Ok(*e.downcast::<RefError>()?),
            }
        };
        let head = |model: &str, arch: &str| {
            format!("# model\t{model}\n# build\t{build}\n# arch\t{arch}\n# complete\t1\t0\n")
        };
        write(&head(&runs, "deepseek41"))?;
        assert!(o.open_named(set).is_ok());
        match refusal(head("/models/other/x.gguf", "deepseek41"))? {
            RefError::Stale {
                dumped_from,
                runs: r,
                ..
            } => assert_eq!(
                (dumped_from.as_str(), r.as_str()),
                ("/models/other/x.gguf", runs.as_str())
            ),
            e => panic!("another file: {e}"),
        }
        match refusal(head(&runs, "deepseek2"))? {
            RefError::Foreign { got, want, .. } => {
                assert_eq!((got.as_str(), want.as_str()), ("deepseek2", "deepseek41"))
            }
            e => panic!("another architecture: {e}"),
        }
        assert!(matches!(
            refusal(head(&runs, "deepseek41").replace("# complete\t1\t0\n", ""))?,
            RefError::Unfinished { .. }
        ));
        assert!(matches!(
            refusal(String::new())?,
            RefError::Unfinished { .. }
        ));
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
}
