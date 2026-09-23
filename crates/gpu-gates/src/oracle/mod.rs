//! The oracle table, one per architecture: which reference dump sets the
//! gates read and the tap names more than one gate asks their manifests for.
//! These are values that differ per architecture, so each is a row here and
//! not a literal in each binary (docs/arch-split.md, 「오라클·탭」); the
//! harness in `lib.rs` that reads a set stays shared.

pub mod deepseek2;
pub mod deepseek41;

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
        let man = RefManifest::read(&ref_dir_named(self.set_name(which)?))?;
        self.check_arch(&man)?;
        Ok(man)
    }

    /// Open the set named `set` under the data directory — a decode-step set
    /// of [`step_sets`](Self::step_sets), say. Its `# arch` line must name
    /// this table's architecture: every set opened by name was dumped after
    /// the dumper wrote that line, so one without it is not the set named.
    pub fn open_named(&self, set: &str) -> Result<RefManifest, GateError> {
        let man = RefManifest::read(&ref_dir_named(set))?;
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
    }
}

#[cfg(test)]
mod tests {
    use super::for_arch;
    use crate::{GateError, RefManifest};
    use model::arch::Arch;

    /// A set opened through the table is of the table's architecture: a
    /// manifest naming another is an error naming both. One without the
    /// `# arch` line is accepted by `open` — a set dumped before the dumper
    /// wrote it — and refused by `open_named`.
    #[test]
    fn a_set_of_another_architecture_is_refused() -> Result<(), GateError> {
        let man = |arch: Option<&str>| RefManifest {
            dir: "/data/set".into(),
            arch: arch.map(str::to_string),
            build: None,
            complete: None,
            header: Default::default(),
            tensors: Vec::new(),
            inputs: Vec::new(),
            ints: Vec::new(),
            skipped_nodes: 0,
            skipped_inputs: 0,
            index: Default::default(),
        };
        let o = for_arch(Arch::Deepseek41).unwrap();
        let err = o
            .check_arch(&man(Some("deepseek2")))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("deepseek2") && err.contains("deepseek41"),
            "{err}"
        );
        assert!(o.check_arch(&man(Some("deepseek41"))).is_ok());
        assert!(o.check_arch(&man(None)).is_ok());

        // An absolute set name is the directory itself.
        let dir = std::env::temp_dir().join(format!("bloomery-open-named-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let set = dir.to_str().ok_or("temp dir is not UTF-8")?;
        let open = |arch_line: &str| -> Result<RefManifest, GateError> {
            let row = "tensor\tx\t0\tf32\t1\t1\t1\t1\t4\t0\tNONE";
            std::fs::write(dir.join("MANIFEST.tsv"), format!("{arch_line}{row}\n"))?;
            o.open_named(set)
        };
        assert!(open("# arch\tdeepseek41\n").is_ok());
        let err = open("# arch\tdeepseek2\n").unwrap_err().to_string();
        assert!(
            err.contains("deepseek2") && err.contains("deepseek41"),
            "{err}"
        );
        assert!(open("").is_err());
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
}
