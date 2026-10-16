//! Host-side dispatch of the single-card Qwen generative seat. A dense
//! `qwen35` file uses the existing Body35; decision layouts keep their own
//! seat, even when they share that backbone.

use crate::GateError;
use model::arch::models::shape::gqa_row;

/// The file architectures that use the Qwen generative seat's two bodies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Qwen3moe,
    Qwen35moe,
    Qwen35,
}

impl Family {
    /// The file's own architecture; decision layouts are not generative.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "qwen3moe" => Some(Self::Qwen3moe),
            "qwen35moe" => Some(Self::Qwen35moe),
            "qwen35" => Some(Self::Qwen35),
            _ => None,
        }
    }

    /// The architecture printed in load and listening records.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Qwen3moe => "qwen3moe",
            Self::Qwen35moe => "qwen35moe",
            Self::Qwen35 => "qwen35",
        }
    }
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
    use super::{Family, decode_mma};

    #[test]
    fn dense_generation_keeps_decision_layouts_out() {
        assert_eq!(Family::from_name("qwen35"), Some(Family::Qwen35));
        for name in ["clef", "qwen35c", "lev", "qwen4exp", "llama", ""] {
            assert_eq!(Family::from_name(name), None, "{name}");
        }
        for name in ["qwen3moe", "qwen35moe", "qwen35"] {
            assert_eq!(Family::from_name(name).unwrap().name(), name);
        }
    }

    #[test]
    fn head_shape_selects_the_existing_flash() {
        assert!(!decode_mma(256, 6).unwrap());
        assert!(decode_mma(256, 8).unwrap());
        assert!(decode_mma(256, 4).unwrap());
        assert!(decode_mma(256, 3).is_err());
    }
}
