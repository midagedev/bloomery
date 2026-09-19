//! Dense FFN (block 0 only in this model): `ffn_norm-0` in, `ffn_out-0` out.
//!
//! Gate: `crates/model/tests/ffn.rs` against the oracle. Owned by the ffn round.
use crate::{ModelError, Tensor2};
use gguf::Gguf;

pub fn dense_ffn(_gguf: &Gguf, _block: usize, _x: &Tensor2) -> Result<Tensor2, ModelError> {
    todo!("ffn round")
}
