//! Block-0 MLA attention: `attn_norm-N` in, `kqv_out-N` out.
//!
//! Gate: `crates/model/tests/attn.rs` against the oracle. Owned by the attn round.
use crate::{ModelError, Tensor2};
use gguf::Gguf;

/// One block's attention. `x` is the block's `attn_norm-N` (already normed).
pub fn block_attn(
    _gguf: &Gguf,
    _block: usize,
    _x: &Tensor2,
    _slots: &[crate::Slot],
) -> Result<Tensor2, ModelError> {
    todo!("attn round")
}
