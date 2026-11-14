//! The header facts of the architectures whose files the common readers take
//! facts from under more than one spelling: one file a model, looked up by the
//! file's `general.architecture`. The mechanism is [`crate::facts`] and
//! [`crate::engram`]; a model adds its table here.

use crate::GgmlType;
use crate::engram::EngramFacts;

pub mod deepseek41;
pub mod exaone_moe;
pub mod kimi_k3;

/// The Engram facts of architecture `arch`; `None` for an architecture whose
/// files carry no Engram table.
#[must_use]
pub fn engram_facts(arch: &str) -> Option<&'static EngramFacts> {
    match arch {
        deepseek41::ARCH => Some(&deepseek41::ENGRAM),
        _ => None,
    }
}

/// The types a tensor of stem `stem` may carry in a file of architecture `arch`
/// (the stem as that architecture's own table spells it); `None` for a stem no
/// file of the architecture holds and for an architecture whose files carry no
/// type table, whose every type is then the classifier's to judge, not a
/// table's. A reader that serves more than one architecture holds every tensor
/// to the file's own table through this one call, so no reader grows a
/// per-architecture branch.
#[must_use]
pub fn allowed_types(arch: &str, stem: &str) -> Option<&'static [GgmlType]> {
    match arch {
        exaone_moe::ARCH => exaone_moe::allowed_types(stem),
        kimi_k3::ARCH | kimi_k3::ARCH_K31 => kimi_k3::allowed_types(stem),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{allowed_types, exaone_moe, kimi_k3};

    /// The one call routes each architecture word to its own stem table (both Kimi words to the same one) and holds
    /// an architecture with no table, or a stem it lacks, to `None`.
    #[test]
    fn allowed_types_routes_by_architecture_word() {
        let stem = "ffn_gate_exps.weight";
        assert_eq!(
            allowed_types(kimi_k3::ARCH, stem),
            kimi_k3::allowed_types(stem)
        );
        assert_eq!(
            allowed_types(kimi_k3::ARCH_K31, stem),
            kimi_k3::allowed_types(stem)
        );
        assert!(allowed_types(kimi_k3::ARCH, stem).is_some());
        assert_eq!(
            allowed_types(exaone_moe::ARCH, stem),
            exaone_moe::allowed_types(stem)
        );
        assert_eq!(allowed_types("no-such-arch", stem), None);
        assert_eq!(allowed_types(kimi_k3::ARCH, "no.such.stem"), None);
    }
}
