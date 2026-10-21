//! The role of every tensor in a `gemma-embedding2` file, by exact name.
//! Every layer carries the same stems, all of them required (llama.cpp
//! creates each with no `TENSOR_NOT_REQUIRED`). The per-layer input's
//! weights are dense weights every token reads in full. A name the table
//! does not hold fails the file with every such name listed.

use gguf::Split;

use super::hparams::Hparams;
use super::names::{
    INP_GATE, LAYER_OUTPUT_SCALE, PER_LAYER_MODEL_PROJ, PER_LAYER_PROJ_NORM, POST_NORM, PROJ,
};
use crate::arch::classify::{Counts, classify_with};
use crate::arch::stems::{Stem, req};
use crate::placement::{ModelTensors, PlacementError, Role};

/// A layer's stems.
const EXTRA: &[Stem] = &[
    req(INP_GATE, Role::EngramDense),
    req(PROJ, Role::EngramDense),
    req(POST_NORM, Role::EngramDense),
    req(LAYER_OUTPUT_SCALE, Role::DenseFfn),
];

/// A tensor's role and layer by its name; `None` when no table holds it.
fn role(name: &str, hp: &Hparams) -> Option<(Role, Option<usize>)> {
    match name {
        "token_embd.weight" => return Some((Role::TokenEmbedding, None)),
        PER_LAYER_MODEL_PROJ | PER_LAYER_PROJ_NORM => return Some((Role::EngramDense, None)),
        "output_norm.weight" | "output.weight" => return Some((Role::Head, None)),
        _ => {}
    }
    crate::arch::gemma_encoder::layer_role(name, hp.n_layer, EXTRA)
}

/// Every tensor of `split` with its role and header facts.
pub fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    classify_with(split, |name| role(name, hp), Counts::new(hp.n_layer, 0, 0))
}
