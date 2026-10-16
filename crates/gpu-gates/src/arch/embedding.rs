//! The embedding architectures a side seat serves, with each model's input bound.

use crate::side_seat::Row;

#[cfg(feature = "gemma-embedding2")]
use crate::side_seat::gemma_embedding2::{bytes, open};
#[cfg(not(feature = "gemma-embedding2"))]
use crate::side_seat::{no_encoder_bytes as bytes, no_encoder_open as open};

/// The architectures a side seat serves.
pub const ROWS: &[Row] = &[
    // The file's context_length is 262144; the model card names 8192 tokens.
    Row {
        arch: model::arch::GEMMA_EMBEDDING2,
        ctx: model::arch::gemma_embedding2::hparams::CARD_CTX,
        bytes,
        open,
        projector_file: Some("mmproj-embeddinggemma-2-BF16.gguf"),
        projector_types: &[vision::arch::gemma4v::PROJECTOR_TYPE],
    },
];

#[cfg(feature = "gemma-embedding2")]
pub(super) fn gemma_embedding2_dims(split: &gguf::Split) -> Result<Vec<usize>, crate::GateError> {
    const MRL: [usize; 3] = [128, 256, 512];
    let n_out = model::arch::gemma_embedding2::hparams::Hparams::read(split)?.n_out;
    if n_out < MRL[0] {
        return Err(format!(
            "embedding model: embedding_length_out {n_out} cannot serve the {}-wide MRL cut",
            MRL[0]
        )
        .into());
    }
    let mut dims: Vec<_> = MRL.into_iter().filter(|&width| width < n_out).collect();
    dims.push(n_out);
    Ok(dims)
}
