//! bloomery-chain: the chain plans the device backends share, as host data, and the walks that
//! order a chain's ops.
//!
//! What a chain of layers runs — each layer's mixer and FFN, every weight's name and type, the
//! geometry the kernels are built for — is decided once from the file's header and the model
//! description, by code that touches no device. The order of a layer's parts is the family's walk
//! (`arch::<family>::walk`) over the ops a backend implements (`arch::<family>::ops`). A backend
//! reads a plan and enqueues the ops the walk asks for; it never decides a layer's kind. This
//! crate holds the plans, the readers that make them and the walks, so a backend that cannot build
//! the CUDA crates (a Mac's) takes the same plan and order the card does.
//!
//! The host rules a description's parts are held to are here too, one reading of each that every
//! backend's gates compare their kernels with: the rope's table ([`rope`]), the conv ring and the
//! short conv ([`linear`], [`shortconv`]), the output gate ([`out_gate`]) and the sandwich residual
//! ([`residual`]).
//!
//! The crate holds no device code and makes no operating-system call of its own: it depends on the
//! model description, the GGUF header reader and the runtime's device-free layer scheduler
//! (`runtime::sched`).
//!
//! A plan's refusals are [`ChainError`]s. A backend converts one into its own error type and keeps
//! the text, so what a gate pins reads the same on every backend.

pub mod arch;
pub mod linear;
pub mod out_gate;
pub mod residual;
pub mod rope;
pub mod router;
pub mod shortconv;
pub mod site;

use std::fmt;

/// A plan refused by name: the entry point that read it and what it could not take.
#[derive(Debug)]
pub enum ChainError {
    /// A geometry or type contract broken: `what` is the entry point, `detail` the lengths, shapes
    /// or types that disagree.
    Shape { what: &'static str, detail: String },
    /// A tensor `what` looked up is absent from the file; `need` completes "is not …".
    Tensor {
        what: &'static str,
        name: String,
        need: &'static str,
    },
}

impl ChainError {
    /// A contract broken by the caller of `what`.
    pub fn shape(what: &'static str, detail: impl Into<String>) -> ChainError {
        ChainError::Shape {
            what,
            detail: detail.into(),
        }
    }

    /// A tensor `what` looked up that is absent; `need` completes "is not …".
    pub fn tensor(what: &'static str, name: impl Into<String>, need: &'static str) -> ChainError {
        ChainError::Tensor {
            what,
            name: name.into(),
            need,
        }
    }
}

impl fmt::Display for ChainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChainError::Shape { what, detail } => write!(f, "{what}: {detail}"),
            ChainError::Tensor { what, name, need } => write!(f, "{what}: {name} is not {need}"),
        }
    }
}

impl std::error::Error for ChainError {}

#[cfg(test)]
mod tests {
    use super::ChainError;

    /// The two refusals read as the card crate's `Shape` and `Tensor` errors do: that crate converts
    /// one into the other with the text unchanged.
    #[test]
    fn a_refusal_reads_as_what_then_its_detail() {
        assert_eq!(
            ChainError::shape("qwen35moe plan", "layer 3: no kernel").to_string(),
            "qwen35moe plan: layer 3: no kernel"
        );
        assert_eq!(
            ChainError::tensor("w", "token_embd.weight", "in the file").to_string(),
            "w: token_embd.weight is not in the file"
        );
    }
}
