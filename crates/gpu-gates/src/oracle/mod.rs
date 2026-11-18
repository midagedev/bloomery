//! The oracle table, one per architecture: which reference dump sets the
//! gates read and the tap names more than one gate asks their manifests for.
//! These are values that differ per architecture, so each is a row here and
//! not a literal in each binary (docs/arch-split.md, 「오라클·탭」); the
//! reader of a set (`refset::ik`) stays shared.

pub mod deepseek2;
#[cfg(test)]
mod deepseek41;
pub mod qwen3moe;

use crate::{GateError, RefManifest, ref_dir_named};
use model::arch::Arch;

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

    /// Open set `which`: its whole manifest ([`RefManifest::read`]) from the
    /// data directory. A manifest whose `# arch` line names another
    /// architecture is an error naming both; one without the line (a set
    /// dumped before the dumper wrote it) is accepted.
    pub fn open(&self, which: Set) -> Result<RefManifest, GateError> {
        let dir = ref_dir_named(self.set_name(which)?);
        let man = RefManifest::read(&dir)?;
        self.check_arch(&man)?;
        Ok(man)
    }

    /// Open the set named `set` under the data directory — a decode-step set
    /// of [`step_sets`](Self::step_sets), say. Its `# arch` line must name
    /// this table's architecture: every set opened by name was dumped after
    /// the dumper wrote that line, so one without it is not the set named.
    pub fn open_named(&self, set: &str) -> Result<RefManifest, GateError> {
        let dir = ref_dir_named(set);
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

/// The table of architecture `a`, its row in the registry below, keyed by the architecture string
/// ([`Arch::name`]). An architecture with no table is an error naming it and why; one with no row
/// is an error naming it.
pub fn for_arch(a: Arch) -> Result<&'static Oracle, GateError> {
    /// One architecture's row: its table, or why it has none.
    enum Row {
        Table(&'static Oracle),
        None(&'static str),
    }
    static ROWS: &[(&str, Row)] = &[
        ("deepseek2", Row::Table(&deepseek2::ORACLE)),
        (
            "deepseek41",
            Row::None("its gates read the fixture file's sets through refset's fixture family"),
        ),
        ("qwen3moe", Row::Table(&qwen3moe::ORACLE)),
        (
            "qwen35moe",
            Row::None("its gate reads the sets through refset's family"),
        ),
        (
            "glm5next",
            Row::None("its gate reads the sets through refset's family"),
        ),
        (
            "mimo2",
            Row::None("its gate reads the sets through refset's family"),
        ),
        ("nemotron_h", Row::None("no program runs it")),
        ("nemotron_h_moe", Row::None("no program runs it")),
        ("gemma-embedding", Row::None("no program runs it")),
        (
            "gemma-embedding2",
            Row::None("its gate reads mainline's sets by its family"),
        ),
    ];
    match ROWS.iter().find(|(name, _)| *name == a.name()) {
        Some(&(_, Row::Table(oracle))) => Ok(oracle),
        Some(&(name, Row::None(why))) => Err(format!("{name} has no oracle row: {why}").into()),
        None => Err(format!("{} has no oracle row", a.name()).into()),
    }
}

#[cfg(test)]
mod tests {
    use super::for_arch;
    use crate::{GateError, RefManifest};
    use model::arch::Arch;

    /// A set opened through the table is of the table's architecture: a
    /// manifest naming another architecture is an error naming both; one
    /// without the `# arch` line is accepted by `open` — a set dumped before
    /// the dumper wrote it — and refused by `open_named`.
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

        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
}
