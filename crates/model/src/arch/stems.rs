//! The role tables' row, shared by the families whose tables name their
//! tensors by stem (`glm5next`, `mimo2`): what a layer of some kind carries,
//! and which of it the file must carry. The tables themselves stay in each
//! family's `roles` — the GGUF names are its architecture's strings — as
//! `classify_with`'s contract says.

use crate::placement::Role;

/// A tensor a layer of some kind carries, by stem: its role, and whether
/// the file must carry it (llama.cpp creates it without
/// `TENSOR_NOT_REQUIRED`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Stem {
    pub(crate) name: &'static str,
    pub(crate) role: Role,
    pub(crate) required: bool,
}

/// A stem the file must carry.
pub(crate) const fn req(name: &'static str, role: Role) -> Stem {
    Stem {
        name,
        role,
        required: true,
    }
}

/// A stem the file may carry.
pub(crate) const fn opt(name: &'static str, role: Role) -> Stem {
    Stem {
        name,
        role,
        required: false,
    }
}

/// The names of `stems` the file must carry.
pub(crate) fn required(stems: &[Stem]) -> impl Iterator<Item = &'static str> + '_ {
    stems.iter().filter(|s| s.required).map(|s| s.name)
}
