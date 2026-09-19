//! Output head: `l_out-26` in, `result_norm` then `result_output` out.
//!
//! Gate: `crates/model/tests/head.rs` against the oracle. Owned by the head round.
use crate::{ModelError, Tensor2};
use gguf::Gguf;

/// Returns the logits for every token position; the oracle only holds the last one.
pub fn head(_gguf: &Gguf, _x: &Tensor2) -> Result<Tensor2, ModelError> {
    todo!("head round")
}
