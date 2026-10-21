//! Every tensor's role, with the shared Gemma encoder trunk as its owner.

use gguf::Split;

use super::hparams::Hparams;
use super::names::{DENSE_2, DENSE_3};
use crate::arch::classify::{Counts, classify_with};
use crate::placement::{ModelTensors, PlacementError, Role};

/// Classify the file; a tensor outside the trunk and dense head is an error.
pub fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    classify_with(
        split,
        |name| match name {
            "token_embd.weight" => Some((Role::TokenEmbedding, None)),
            "output_norm.weight" | "output.weight" | DENSE_2 | DENSE_3 => Some((Role::Head, None)),
            _ => crate::arch::gemma_encoder::layer_role(name, hp.n_layer, &[]),
        },
        Counts::new(hp.n_layer, 0, 0),
    )
}
