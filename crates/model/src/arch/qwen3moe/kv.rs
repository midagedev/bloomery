//! The KV bytes a Qwen3-MoE layer holds at `ctx_max` tokens — placement's
//! accounting, not the cache itself.
//!
//! Grouped-query attention: every layer keeps its own K and V planes, each
//! `ctx_max` rows of `n_head_kv · head_dim` values in f16. No layer shares a
//! plane and none keeps a window, so every layer holds the same bytes.

use gguf::Split;

use super::hparams::Hparams;
use super::names;
use crate::placement::{KvBytes, PlacementError};

const F16_BYTES: u64 = 2;

/// One file's KV byte function.
#[derive(Clone, Copy, Debug)]
pub struct KvLayout {
    /// Values in one position's key row, which the value row matches:
    /// `n_head_kv · head_dim`.
    row: u64,
}

impl KvLayout {
    /// The row width from `hp`, checked against every layer's `attn_k` and
    /// `attn_v` output width: a projection that does not write that row is
    /// an error naming it.
    pub fn from_file(split: &Split, hp: &Hparams) -> Result<KvLayout, PlacementError> {
        let row = (hp.n_head_kv * hp.head_dim) as u64;
        for l in 0..hp.n_layer {
            for name in [names::attn_k(l), names::attn_v(l)] {
                let Some((_, t)) = split.find(&name) else {
                    return Err(PlacementError::Tensor {
                        name,
                        detail: "is not in the file".to_string(),
                    });
                };
                if t.dims.get(1) != Some(&row) {
                    return Err(PlacementError::Tensor {
                        name,
                        detail: format!(
                            "has dims {:?}; its rows are {} key heads of {}",
                            t.dims, hp.n_head_kv, hp.head_dim
                        ),
                    });
                }
            }
        }
        Ok(KvLayout { row })
    }

    /// Bytes of one position of one layer: its key row and its value row.
    #[must_use]
    pub fn position_bytes(&self) -> u64 {
        2 * self.row * F16_BYTES
    }
}

impl KvBytes for KvLayout {
    fn layer_bytes(&self, _layer: usize, ctx_max: u64) -> u64 {
        ctx_max * self.position_bytes()
    }
}
