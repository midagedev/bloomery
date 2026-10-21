//! The role of every tensor in a Qwen3-MoE file, by exact name: the three
//! model-level names, then the per-layer stems of [`names::LAYER`]. There is
//! no prefix rule and no catch-all: a name the table does not hold — a merged
//! `attn_qkv`, a dense `ffn_up`, a shared expert, a router bias — fails the
//! file with every such name listed, because this chain would not read it.

use gguf::Split;

use super::hparams::Hparams;
use crate::arch::classify::{Counts, classify_groups};
use crate::arch::stems::{ATTENTION, MOE_ROUTER, MOE_STACKS, Stem, join};
use crate::placement::{ModelTensors, PlacementError};

/// The stems every layer carries and nothing else: the attention block with its per-head norms, the
/// pre-FFN norm, the router and the routed stacks ([`names::LAYER`], in its order).
pub(super) const STEMS: &[Stem] = &join::<12>(&[ATTENTION, MOE_ROUTER, MOE_STACKS]);

/// Every tensor of `split` with its role and header facts, and the model's
/// layer and expert counts from `hp`, the file's [`Hparams`].
pub fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    let e = &hp.experts;
    classify_groups(
        split,
        &[STEMS],
        Counts::new(hp.n_layer, e.n_expert, e.n_used),
    )
}

#[cfg(test)]
mod tests {
    use super::STEMS;
    use crate::arch::classify::role_in;
    use crate::arch::qwen3moe::names;
    use crate::placement::Role;

    /// A tensor's role and layer by its name over the table.
    fn role(name: &str, n_layer: usize) -> Option<(Role, Option<usize>)> {
        role_in(&[STEMS], name, n_layer)
    }

    /// The table is [`names::LAYER`] with every stem required; every name of it has its role on
    /// its layer; a name the table does not hold, or a layer past the model, has none, so
    /// `classify` lists it. Mutant: a stem dropped, or the layer bound `<=`.
    #[test]
    fn the_table_is_exact() {
        assert!(STEMS.iter().map(|s| s.name).eq(names::LAYER));
        assert!(STEMS.iter().all(|s| s.required));
        assert_eq!(
            role("token_embd.weight", 48),
            Some((Role::TokenEmbedding, None))
        );
        assert_eq!(role("output.weight", 48), Some((Role::Head, None)));
        assert_eq!(role("output_norm.weight", 48), Some((Role::Head, None)));
        for s in STEMS {
            let name = format!("blk.47.{}", s.name);
            assert_eq!(role(&name, 48), Some((s.role, Some(47))), "{name}");
        }
        assert_eq!(
            role("blk.47.ffn_norm.weight", 48),
            Some((Role::FfnNorm, Some(47)))
        );
        assert_eq!(
            role("blk.47.ffn_gate_inp.weight", 48),
            Some((Role::Router, Some(47)))
        );
        assert_eq!(
            role("blk.47.attn_q.weight", 48),
            Some((Role::Attention, Some(47)))
        );
        for name in [
            "blk.48.attn_q.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_up_shexp.weight",
            "blk.0.exp_probs_b.bias",
            "blk.0.attn_q.bias",
            "rope_freqs.weight",
            "output.bias",
        ] {
            assert_eq!(role(name, 48), None, "{name}");
        }
    }
}
