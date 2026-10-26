//! Host-side dispatch of the single-card Qwen generative seat. The seat takes a file by the program
//! that runs its description's layers (`placement::accepts::program_of`), never by its
//! architecture string: a dense trunk the Qwen3.5 body runs uses the existing Body35, whatever the
//! trunk is called. A file that declares a decision type keeps its own seat, even when it shares
//! that backbone.

use crate::GateError;
use crate::model_file::refuse_decision_model;
use gguf::Split;
use model::arch::models::shape::gqa_row;
use model::arch::models::{Ffn, ModelSpec};
use model::placement::accepts::{Program, program_of};

/// The families of the Qwen generative seat's two bodies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// The qwen3moe body's files.
    Qwen3moe,
    /// The Qwen3.5 body on a trunk with a routed feed-forward block.
    Qwen35moe,
    /// The Qwen3.5 body on a trunk whose every feed-forward block is dense.
    Qwen35,
}

impl Family {
    /// The seat's family of the file `split` holds: its description's program, and for the
    /// Qwen3.5 body whether every feed-forward block is dense. A file that declares a decision
    /// type (`<arch>.decision.type`: Clef's layout carries its head beside the trunk), a file with
    /// no description and one no program of the seat runs are refused by name.
    ///
    /// # Errors
    /// The refusal, naming the file's architecture and why the seat takes none of it.
    pub fn of(split: &Split) -> Result<Self, GateError> {
        let arch = split.architecture().unwrap_or("<missing>");
        refuse_decision_model(split)?;
        let spec = model::arch::spec(split)
            .map_err(|e| format!("a {arch} file has no description to serve: {e}"))?
            .spec;
        let program = program_of(spec.arch);
        Self::of_spec(&spec).ok_or_else(|| {
            format!(
                "a {arch} file is run by {}, and the Qwen generative seat serves the files of \
                 the {} and {} programs",
                program.map_or("no program", Program::name),
                Program::Qwen3moeBody.name(),
                Program::Qwen35Body.name()
            )
            .into()
        })
    }

    /// The seat's family of the description `spec`: its program, and for the Qwen3.5 body whether
    /// every feed-forward block is dense ([`Family::of`] reads a file's); `None` for a description
    /// no program of the seat runs.
    #[must_use]
    pub fn of_spec(spec: &ModelSpec) -> Option<Self> {
        Self::pick(program_of(spec.arch), dense(spec))
    }

    /// The family of a description whose layers `program` runs, `dense` its feed-forward blocks.
    fn pick(program: Option<Program>, dense: bool) -> Option<Self> {
        match program? {
            Program::Qwen3moeBody => Some(Self::Qwen3moe),
            Program::Qwen35Body if dense => Some(Self::Qwen35),
            Program::Qwen35Body => Some(Self::Qwen35moe),
            _ => None,
        }
    }

    /// The family's word, for refusals that name what the seat serves; the records print the
    /// file's own architecture string.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Qwen3moe => "qwen3moe",
            Self::Qwen35moe => "qwen35moe",
            Self::Qwen35 => "qwen35",
        }
    }
}

/// Whether every layer's feed-forward block of `spec` is dense: a trunk with no routed expert.
fn dense(spec: &ModelSpec) -> bool {
    spec.layers
        .iter()
        .all(|l| matches!(l.ffn, Ffn::Dense { .. }))
}

/// Body35's decode flash from the file's head shape. The existing pair
/// kernel (27B: 24 query heads / 4 KV heads) has a scalar pass only; the
/// group-of-eight and quad entries retain the tensor-core default.
pub fn decode_mma(head: usize, group: usize) -> Result<bool, GateError> {
    let row = gqa_row(u32::try_from(head)?, u32::try_from(group)?)
        .ok_or_else(|| format!("no Qwen decode flash for head {head}, group {group}"))?;
    Ok(row.pack != 2)
}

#[cfg(test)]
mod tests {
    use super::{Family, Program, decode_mma};
    use crate::model_file::test_header;
    use gguf::Value;

    /// A file of Clef's layout carries its decision type in its header, and the seat refuses it
    /// before it reads a description.
    #[test]
    fn a_file_that_declares_a_decision_type_is_not_the_seats() {
        let keys = vec![(
            "clef.decision.type".to_owned(),
            Value::String("clef".into()),
        )];
        let path = std::env::temp_dir().join(format!(
            "bloomery-qwen-seat-decision-{}.gguf",
            std::process::id()
        ));
        let split = test_header(&path, "clef", keys);
        let err = Family::of(&split).unwrap_err().to_string();
        drop(split);
        std::fs::remove_file(path).unwrap();
        assert_eq!(
            err,
            "a clef file declares clef.decision.type = clef: a decision model is the decide seat's, not generation"
        );
    }

    #[test]
    fn the_seat_takes_a_program_and_leaves_the_others() {
        let by = |p, dense| Family::pick(Some(p), dense);
        assert_eq!(by(Program::Qwen3moeBody, false), Some(Family::Qwen3moe));
        assert_eq!(by(Program::Qwen35Body, false), Some(Family::Qwen35moe));
        assert_eq!(by(Program::Qwen35Body, true), Some(Family::Qwen35));
        for other in [
            Program::Deepseek41Chain,
            Program::Qwen38Body,
            Program::Glm5nextBody,
            Program::NemotronHBody,
        ] {
            assert_eq!(by(other, false), None, "{}", other.name());
        }
        assert_eq!(Family::pick(None, true), None);
    }

    #[test]
    fn head_shape_selects_the_existing_flash() {
        assert!(!decode_mma(256, 6).unwrap());
        assert!(decode_mma(256, 8).unwrap());
        assert!(decode_mma(256, 4).unwrap());
        assert!(decode_mma(256, 3).is_err());
    }
}
