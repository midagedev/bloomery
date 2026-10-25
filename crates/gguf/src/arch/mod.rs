//! The header facts of the architectures whose files the common readers take
//! facts from under more than one spelling: one file a model, looked up by the
//! file's `general.architecture`. The mechanism is [`crate::facts`] and
//! [`crate::engram`]; a model adds its table here.

use crate::engram::EngramFacts;

pub mod deepseek41;

/// The Engram facts of architecture `arch`; `None` for an architecture whose
/// files carry no Engram table.
#[must_use]
pub fn engram_facts(arch: &str) -> Option<&'static EngramFacts> {
    match arch {
        deepseek41::ARCH => Some(&deepseek41::ENGRAM),
        _ => None,
    }
}
