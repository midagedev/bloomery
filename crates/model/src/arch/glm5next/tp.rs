//! How tensor parallel splits each glm5next tensor ([`cut`]): along the dim and by the unit a
//! rank's part holds whole there, or whole on every rank. The dims are ggml's order, the file's
//! and the checkpoint map's alike (`model_hf::arch::glm5next`), so one rule serves the GGUF rank
//! load (`gpu-glm5next`'s `tp`) and the checkpoint's plan; the plan of the cuts is
//! [`plan_tp`](crate::placement::tp::plan_tp)'s.
//!
//! KDA and latent attention split by head: their q, k, v, gate and beta rows, the conv channels,
//! the decay and the latent heads' `q_b`, `k_b` and `v_b`, and the output projection along its
//! input. Dense blocks, the shared expert and every routed expert split by inner width: gate and
//! up by row, down along its input. The head splits by vocabulary row. The norms, the low-rank
//! inputs every head shares (`ssm_f_a`, `ssm_g_a`, `attn_q_a`, `attn_kv_a_mqa`), the indexer, the
//! router and its bias, the hyper-connection mixes, the next-token layer's own tensors and the
//! embedding stay whole on every rank. A rank's load takes the trunk's layers, not the next-token
//! layers past them ([`in_trunk`]), and holds them on the card but the embedding, which is read
//! on the host ([`card_holds`]).
//!
//! A part keeps whole blocks of the tensor's card type, and along K at least its role's kernel
//! unit ([`cut_of`]); which indices a rank holds is then [`Cut::span`]'s ([`rank_span`]), for the
//! GGUF file and the checkpoint folder alike.

use std::ops::Range;

use gguf::GgmlType;

use super::{names, roles};
use crate::placement::PlacementError;
use crate::placement::tp::{self, Cut};

/// Whether a tensor of layer `layer` (`None` for one of no layer) is one a TP rank of a trunk of
/// `layers` layers loads: not one of the next-token layers past the trunk's.
pub fn in_trunk(layer: Option<usize>, layers: usize) -> bool {
    layer.is_none_or(|l| l < layers)
}

/// Whether a TP rank of a trunk of `layers` layers holds the tensor `name` of layer `layer` on the
/// card: one it loads ([`in_trunk`]) but the token embedding, which stays in the host file.
pub fn card_holds(name: &str, layer: Option<usize>, layers: usize) -> bool {
    name != names::token_embd() && in_trunk(layer, layers)
}

/// Where the glm5next tensor `name` of `dims` splits among ranks, `None` for whole on every rank.
/// A head's part along a dim is that dim over `heads`. A stem with no rule, a split tensor of
/// other dims than its stem has, and a head split of an extent that is not whole heads are
/// refused, the reason the error.
pub fn cut(name: &str, dims: &[u64], heads: u64) -> Result<Option<Cut>, PlacementError> {
    let stem = name
        .strip_prefix("blk.")
        .and_then(|r| r.split_once('.'))
        .map_or(name, |(_, stem)| stem);
    // The dims the tensor has, the dim it splits along, and whether by head or by one value.
    let (rank, dim, by_head) = match stem {
        "token_embd.weight"
        | "output_norm.weight"
        | "attn_norm.weight"
        | "ffn_norm.weight"
        | "ssm_f_a.weight"
        | "ssm_g_a.weight"
        | "ssm_norm.weight"
        | "attn_q_a.weight"
        | "attn_q_a_norm.weight"
        | "attn_kv_a_mqa.weight"
        | "attn_kv_a_norm.weight"
        | "ffn_gate_inp.weight"
        | "exp_probs_b.bias" => return Ok(None),
        _ if stem.starts_with("indexer")
            || stem.starts_with("hc_")
            || stem.starts_with("nextn.") =>
        {
            return Ok(None);
        }
        "attn_q.weight" | "attn_k.weight" | "attn_v.weight" | "ssm_f_b.weight"
        | "ssm_g_b.weight" | "ssm_beta.weight" | "attn_q_b.weight" => (2, 1, true),
        "attn_output.weight" => (2, 0, true),
        "ssm_a" | "ssm_dt.bias" => (1, 0, true),
        "ssm_conv1d_q.weight"
        | "ssm_conv1d_k.weight"
        | "ssm_conv1d_v.weight"
        | "attn_k_b.weight"
        | "attn_v_b.weight" => (3, 2, true),
        "ffn_gate.weight"
        | "ffn_up.weight"
        | "ffn_gate_shexp.weight"
        | "ffn_up_shexp.weight"
        | "output.weight" => (2, 1, false),
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" => (3, 1, false),
        "ffn_down.weight" | "ffn_down_shexp.weight" => (2, 0, false),
        "ffn_down_exps.weight" => (3, 0, false),
        _ => {
            return Err(PlacementError::Tensor {
                name: name.into(),
                detail: "no tensor-parallel split rule".into(),
            });
        }
    };
    tp::cut_units(name, dims, rank, dim, by_head.then_some(heads))
}

/// The [`cut`] of the glm5next tensor `name` of `dims` held in card type `ty`, with the values a
/// part keeps whole along dims 0 and 1: whole blocks of the type, along K at least the K unit of
/// its role's kernel ([`card_blocks`]); `None` for whole on every rank. A refusal of [`cut`], a
/// split tensor with no role or of a type with no block, and blocks past u64 are refused by name.
pub fn cut_of(
    name: &str,
    ty: GgmlType,
    dims: &[u64],
    heads: u64,
) -> Result<Option<(Cut, [u64; 2])>, PlacementError> {
    tp::cut_blocks(name, ty, cut(name, dims, heads)?, roles::tensor_role(name))
}

/// Rank `rank` of `ranks`' indices of the glm5next tensor `name` of card type `ty` and `dims`
/// over `heads` heads: the dim its [`cut_of`] splits and the indices along it ([`Cut::span`]);
/// `None` for the whole tensor, at one rank or where the family keeps it whole. A rank that is
/// not one of the ranks, and a refusal of [`cut_of`] or [`Cut::span`], are refused by name.
pub fn rank_span(
    name: &str,
    ty: GgmlType,
    dims: &[u64],
    heads: u64,
    rank: usize,
    ranks: usize,
) -> Result<Option<(usize, Range<u64>)>, PlacementError> {
    tp::span_of(name, dims, rank, ranks, || cut_of(name, ty, dims, heads))
}

#[cfg(test)]
mod tests {
    use super::{Cut, GgmlType, PlacementError, cut, cut_of, rank_span};

    /// Each kind of split at the GLM fixture's dims (64 heads), the whole tensors, and the
    /// refusals: a stem with no rule, a split tensor of another dim count, and a head split of an
    /// extent that is not whole heads.
    #[test]
    fn every_stem_cuts_by_its_rule_and_the_rest_is_refused() {
        let at = |dim, unit| Some(Cut { dim, unit });
        for (name, dims, want) in [
            ("blk.0.attn_q.weight", &[4096u64, 8192][..], at(1, 128)),
            ("blk.0.attn_output.weight", &[8192, 4096], at(0, 128)),
            ("blk.0.ssm_a", &[64], at(0, 1)),
            ("blk.0.ssm_conv1d_v.weight", &[4, 1, 8192], at(2, 128)),
            ("blk.3.attn_q_b.weight", &[1536, 16384], at(1, 256)),
            ("blk.3.attn_k_b.weight", &[256, 512, 64], at(2, 1)),
            ("blk.0.ffn_down.weight", &[12288, 4096], at(0, 1)),
            ("blk.3.ffn_up_exps.weight", &[4096, 512, 288], at(1, 1)),
            ("blk.3.ffn_down_exps.weight", &[512, 4096, 288], at(0, 1)),
            ("output.weight", &[4096, 154880], at(1, 1)),
        ] {
            assert_eq!(cut(name, dims, 64).unwrap(), want, "{name}");
        }
        for name in [
            "token_embd.weight",
            "blk.0.ssm_f_a.weight",
            "blk.3.attn_kv_a_mqa.weight",
            "blk.3.indexer.attn_k.weight",
            "blk.0.hc_attn_fn.weight",
            "blk.6.nextn.eh_proj.weight",
        ] {
            assert_eq!(cut(name, &[4096, 128], 64).unwrap(), None, "{name}");
        }
        assert!(matches!(
            cut("blk.3.attn_sinks.weight", &[64], 64),
            Err(PlacementError::Tensor { name, detail })
                if name == "blk.3.attn_sinks.weight" && detail == "no tensor-parallel split rule"
        ));
        for (name, dims) in [
            ("blk.3.ffn_up_exps.weight", &[4096u64, 288][..]),
            ("blk.0.ffn_down.weight", &[4096, 2048, 2]),
            ("blk.0.ssm_a", &[64, 1]),
        ] {
            assert!(
                matches!(cut(name, dims, 64), Err(PlacementError::Tensor { detail, .. }) if detail.ends_with("-D")),
                "{name}"
            );
        }
        assert!(cut("blk.0.attn_q.weight", &[4096, 8190], 64).is_err());
        assert!(cut("blk.0.attn_q.weight", &[4096, 8192], 0).is_err());
    }

    /// A cut keeps its type's block along K, at least its role's kernel unit there (NVFP4's
    /// 64-value block, 256 for a dense down and 32 for a routed one), and one row along dim 1; a
    /// whole tensor has no blocks, and a type with no block is refused by name.
    #[test]
    fn a_cut_keeps_its_card_type_s_blocks_and_kernel_unit() {
        let nv = GgmlType::NVFP4;
        for (name, dims, want) in [
            ("blk.0.ffn_down.weight", &[12288u64, 4096][..], [256, 1]),
            ("blk.3.ffn_down_exps.weight", &[512, 4096, 288], [64, 1]),
            ("blk.3.ffn_up_exps.weight", &[4096, 512, 288], [64, 1]),
        ] {
            let (_, blocks) = cut_of(name, nv, dims, 64).unwrap().unwrap();
            assert_eq!(blocks, want, "{name}");
        }
        assert_eq!(
            cut_of("blk.0.attn_norm.weight", nv, &[4096], 64).unwrap(),
            None
        );
        let e = cut_of("blk.0.ffn_down.weight", GgmlType::Unknown(99), &[64, 4], 64);
        assert!(
            matches!(e, Err(PlacementError::Tensor { name, .. }) if name == "blk.0.ffn_down.weight")
        );
    }

    /// One rank reads every tensor whole; past one, a split tensor's rank reads its indices along
    /// the cut and a kept one the whole; a rank past the ranks and a cut off the type's block are
    /// refused by name.
    #[test]
    fn a_rank_reads_its_indices_of_each_cut() {
        let nv = GgmlType::NVFP4;
        let up = ("blk.3.ffn_up_exps.weight", &[4096u64, 512, 288][..]);
        let down = ("blk.3.ffn_down_exps.weight", &[512u64, 4096, 288][..]);
        let span =
            |(name, dims): (&str, &[u64]), rank, ranks| rank_span(name, nv, dims, 64, rank, ranks);
        assert_eq!(span(up, 0, 1).unwrap(), None);
        assert_eq!(span(up, 1, 4).unwrap(), Some((1, 128..256)));
        assert_eq!(span(down, 3, 4).unwrap(), Some((0, 384..512)));
        assert_eq!(
            span(("blk.0.attn_norm.weight", &[4096]), 1, 2).unwrap(),
            None
        );
        for (rank, ranks) in [(1, 1), (2, 2), (0, 0)] {
            assert!(span(up, rank, ranks).is_err(), "{rank}/{ranks}");
        }
        let e = span(down, 0, 16).unwrap_err().to_string();
        assert!(e.contains(down.0) && e.contains("64"), "{e}");
    }
}
