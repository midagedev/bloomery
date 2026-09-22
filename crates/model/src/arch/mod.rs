//! Which architecture a file holds, and the only place in this crate that reads
//! `general.architecture`.
//!
//! Variant names are the GGUF strings themselves, not abbreviations: a single
//! `grep deepseek41` has to find the module, the metadata keys, the tool profile
//! and the gates at once.

use crate::ModelError;
use gguf::Gguf;

pub mod deepseek2;

/// The architectures this engine can run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arch {
    Deepseek2,
    Deepseek41,
}

impl Arch {
    /// The architecture a file declares. The error carries the string the file
    /// held, because that string is what the reader has to act on.
    pub fn detect(gguf: &Gguf) -> Result<Arch, ModelError> {
        let name = gguf
            .architecture()
            .ok_or_else(|| ModelError::UnknownArchitecture("<missing>".to_string()))?;
        Arch::from_name(name)
    }

    /// The string → variant step on its own, so it can be tested without a file.
    pub fn from_name(name: &str) -> Result<Arch, ModelError> {
        match name {
            "deepseek2" => Ok(Arch::Deepseek2),
            "deepseek41" => Ok(Arch::Deepseek41),
            other => Err(ModelError::UnknownArchitecture(other.to_string())),
        }
    }

    /// The `general.architecture` string this variant stands for.
    pub fn name(&self) -> &'static str {
        match self {
            Arch::Deepseek2 => "deepseek2",
            Arch::Deepseek41 => "deepseek41",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Arch;

    #[test]
    fn known_names_round_trip() {
        for a in [Arch::Deepseek2, Arch::Deepseek41] {
            assert_eq!(Arch::from_name(a.name()).unwrap(), a);
        }
    }

    #[test]
    fn unknown_name_is_reported_verbatim() {
        let err = Arch::from_name("llama").unwrap_err();
        assert!(
            err.to_string().contains("llama"),
            "the error must name the string the file held, got {err}"
        );
    }
}
