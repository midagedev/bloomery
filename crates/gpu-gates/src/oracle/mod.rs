//! The oracle table, one per architecture: which reference dump sets the
//! gates read and the tap names more than one gate asks their manifests for.
//! These are values that differ per architecture, so each is a row here and
//! not a literal in each binary (docs/arch-split.md, 「오라클·탭」); the
//! harness in `lib.rs` that reads a set stays shared.

pub mod deepseek2;

use crate::GateError;
use model::arch::Arch;

/// One architecture's reference sets and shared tap names. A set name is a
/// directory under the data directory ([`crate::ref_dir_named`]).
pub struct Oracle {
    /// The architecture the sets were dumped from.
    pub arch: Arch,
    /// ik's CUDA dump with the v2 manifest columns and logical twins — the
    /// set [`crate::ref_dir`] resolves to unless the environment says otherwise.
    pub cuda_set: &'static str,
    /// ik's CPU dump.
    pub cpu_set: &'static str,
    /// The pre-v2 CUDA dump: plain files only, no logical twins.
    pub legacy_cuda_set: &'static str,
    /// Tap names asked of a manifest by more than one gate binary. A name
    /// that only one gate reads stays in that gate.
    pub taps: &'static [&'static str],
}

/// The table of architecture `a`; one the engine names but has no reference
/// dump for yet is an error naming it.
pub fn for_arch(a: Arch) -> Result<&'static Oracle, GateError> {
    match a {
        Arch::Deepseek2 => Ok(&deepseek2::ORACLE),
        Arch::Deepseek41 => Err(format!(
            "oracle: no reference sets for architecture {:?} yet",
            a.name()
        )
        .into()),
    }
}
