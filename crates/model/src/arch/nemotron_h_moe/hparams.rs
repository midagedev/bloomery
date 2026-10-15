//! Every `nemotron_h_moe.*` key the description reads, and each layer's kind
//! from the per-layer arrays. A layer is Mamba-2 where both
//! `attention.head_count_kv` and `feed_forward_length` are 0, attention
//! where only the feed-forward length is 0, and a latent MoE layer otherwise
//! (nemotron-h.cpp:12-16, :102, :110-111: with `expert_count` above 0 every
//! feed-forward layer is routed). The layer's marker tensor has to say the
//! same thing. The Mamba-2 and attention keys and tensors are read by the
//! hybrid trunk's common pieces (`arch::hybrid`), as `nemotron_h` reads its.
//!
//! The MoE layers differ from layer to layer: `expert_feed_forward_length`
//! and `expert_used_count` are per-layer arrays, 0 on every other layer, and
//! a layer's expert count is its router's row count, which its stacks' last
//! dim and its bias repeat. `expert_count` is the largest layer's. The
//! routed experts run in a `moe_latent_size` latent, and a shared expert of
//! `expert_shared_feed_forward_length` runs beside them on the hidden row
//! (nemotron-h.cpp:111-131, :282-334).

use gguf::{GgmlType, Split};

use super::names::{
    self, ATTN_NORM, EXP_PROBS_B, FFN_DOWN_EXPS, FFN_DOWN_SHEXP, FFN_GATE_INP, FFN_LATENT_DOWN,
    FFN_LATENT_UP, FFN_UP_EXPS, FFN_UP_SHEXP, SSM_IN_SCALE, SSM_OUT_SCALE,
};
use super::roles;
pub use crate::arch::hybrid::Ssm;
use crate::arch::hybrid::{Attn, attn, layer_kinds, layers_agree, mamba_dims, no_nextn, ssm};
use crate::arch::{
    finite_positive, meta_bool, metadata, n_vocab, one_expert_group, per_layer_count, positive,
    sigmoid_gating, unread_keys_of,
};
use crate::placement::PlacementError;

/// What a layer is, from the two per-layer arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Both arrays 0.
    Mamba,
    /// Feed-forward length 0, KV heads not.
    Attention,
    /// A positive feed-forward length: the latent MoE.
    Moe,
}

impl Kind {
    /// The kind as a refusal names a layer of it.
    #[must_use]
    pub fn what(self) -> &'static str {
        match self {
            Kind::Mamba => "a Mamba-2 layer",
            Kind::Attention => "an attention layer",
            Kind::Moe => "a MoE layer",
        }
    }
}

/// One MoE layer's routed experts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Experts {
    /// The router's rows (`ffn_gate_inp`'s second dim), which the stacks'
    /// last dim and the bias repeat.
    pub n_expert: usize,
    /// `expert_used_count` at the layer.
    pub top_k: usize,
    /// `expert_feed_forward_length` at the layer: one expert's values.
    pub ff: usize,
}

/// The hyperparameters of one `nemotron_h_moe` file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// `block_count`. This reader runs no next-token layer, so every block
    /// is a trunk layer.
    pub n_layer: usize,
    /// `embedding_length`.
    pub n_embd: usize,
    /// `context_length`.
    pub n_ctx: usize,
    /// `vocab_size`, else the token list's length.
    pub n_vocab: usize,
    /// `attention.layer_norm_rms_epsilon`.
    pub rms_eps: f32,
    /// `attention.head_count`.
    pub n_head: usize,
    /// `attention.key_length`.
    pub key_len: usize,
    /// `attention.value_length`.
    pub value_len: usize,
    /// The Mamba-2 widths. One set for every Mamba-2 layer.
    pub ssm: Ssm,
    /// `attention.head_count_kv`, per layer. 0 on a Mamba-2 or MoE layer.
    pub kv: Vec<usize>,
    /// Every layer's kind, in order.
    pub kinds: Vec<Kind>,
    /// `expert_count`: the largest MoE layer's expert count.
    pub n_expert: usize,
    /// Each layer's routed experts; `None` on a Mamba-2 or attention layer.
    pub experts: Vec<Option<Experts>>,
    /// `moe_latent_size`: the values the routed experts read and write.
    pub latent: usize,
    /// `expert_shared_feed_forward_length`: the shared expert's values.
    pub shared_ff: usize,
    /// `expert_weights_norm`: the kept weights renormalized to sum to one.
    pub weights_norm: bool,
    /// `expert_weights_scale`: the factor on the kept weights.
    pub weights_scale: f32,
    /// The key defaults taken, each with its value and why.
    pub defaults: Vec<String>,
}

/// Every `nemotron_h_moe.` key [`Hparams::read`] reads, by suffix.
const READ: &[&str] = &[
    "block_count",
    "nextn_predict_layers",
    "context_length",
    "embedding_length",
    "vocab_size",
    "feed_forward_length",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.layer_norm_rms_epsilon",
    "attention.layer_norm_epsilon",
    "attention.key_length",
    "attention.value_length",
    "rope.dimension_count",
    "ssm.conv_kernel",
    "ssm.state_size",
    "ssm.group_count",
    "ssm.inner_size",
    "ssm.time_step_rank",
    "expert_count",
    "expert_used_count",
    "expert_group_count",
    "expert_group_used_count",
    "expert_gating_func",
    "expert_feed_forward_length",
    "expert_shared_feed_forward_length",
    "expert_shared_count",
    "expert_weights_norm",
    "expert_weights_scale",
    "moe_latent_size",
];

/// The architecture keys and `general.sampling.*` keys of `split` that
/// [`Hparams::read`] does not read: listed, not refused.
#[must_use]
pub fn unread_keys(split: &Split) -> Vec<String> {
    unread_keys_of(split, READ)
}

impl Hparams {
    /// Every value from `split`'s headers. A missing required key, a value
    /// the description cannot hold, a per-layer array of the wrong length, a
    /// layer whose arrays or tensors disagree with its kind, or a tensor of
    /// other dims than the keys give it is an error naming the key or the
    /// tensor.
    pub fn read(split: &Split) -> Result<Hparams, PlacementError> {
        let n_layer = positive(split, "block_count")?;
        no_nextn(split)?;
        let n_embd = positive(split, "embedding_length")?;
        let n_ctx = positive(split, "context_length")?;
        let rms_eps = finite_positive(split, "attention.layer_norm_rms_epsilon")?;
        finite_positive(split, "attention.layer_norm_epsilon")?;
        let attn = attn(split)?;
        let ssm = ssm(split)?;
        let n_vocab = n_vocab(split)?;
        let n_expert = positive(split, "expert_count")?;
        let latent = positive(split, "moe_latent_size")?;
        let shared_ff = positive(split, "expert_shared_feed_forward_length")?;
        let shared = positive(split, "expert_shared_count")?;
        if shared != 1 {
            return Err(metadata(
                split,
                "expert_shared_count",
                format!("is {shared}; a layer runs one shared expert"),
            ));
        }
        // Required: nemotron-h.cpp:312 builds the sigmoid router whatever the
        // key says, so a file that says another is refused, not defaulted.
        let mut defaults = Vec::new();
        sigmoid_gating(split, None, &mut defaults)?;
        let weights_norm = meta_bool(split, "expert_weights_norm")?;
        let weights_scale = finite_positive(split, "expert_weights_scale")?;
        one_expert_group(split, "the nemotron_h_moe router")?;
        let kv = per_layer_count(split, "attention.head_count_kv", n_layer)?;
        let ff = per_layer_count(split, "feed_forward_length", n_layer)?;
        let expert_ff = per_layer_count(split, "expert_feed_forward_length", n_layer)?;
        let top_k = per_layer_count(split, "expert_used_count", n_layer)?;
        let kinds = layer_kinds(
            split,
            &kv,
            &ff,
            [
                (names::SSM_IN, Kind::Mamba),
                (names::ATTN_Q, Kind::Attention),
                (FFN_UP_EXPS, Kind::Moe),
            ],
        )?;
        for (l, &kind) in kinds.iter().enumerate() {
            if kind == Kind::Attention {
                attn.check_layer(split, l, kv[l])?;
            }
            routed_arrays_agree(split, l, kind, expert_ff[l], top_k[l])?;
        }
        layers_agree(
            split,
            &kinds,
            roles::stems_of,
            &[Kind::Mamba, Kind::Attention, Kind::Moe],
        )?;
        let experts = kinds
            .iter()
            .enumerate()
            .map(|(l, &kind)| {
                (kind == Kind::Moe)
                    .then(|| layer_experts(split, l, n_expert, expert_ff[l], top_k[l]))
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let largest = experts.iter().flatten().map(|e| e.n_expert).max();
        if largest != Some(n_expert) {
            return Err(metadata(
                split,
                "expert_count",
                format!(
                    "is {n_expert}, and the largest layer's router has {} rows",
                    largest.map_or_else(|| "no MoE layer, so no".to_string(), |n| n.to_string())
                ),
            ));
        }
        let hp = Hparams {
            n_layer,
            n_embd,
            n_ctx,
            n_vocab,
            rms_eps,
            n_head: attn.n_head,
            key_len: attn.key_len,
            value_len: attn.value_len,
            ssm,
            kv,
            kinds,
            n_expert,
            experts,
            latent,
            shared_ff,
            weights_norm,
            weights_scale,
            defaults,
        };
        crate::arch::hybrid::dims_agree(
            split,
            hp.n_embd,
            hp.n_vocab,
            hp.kinds.iter().enumerate().map(|(l, &k)| hp.want(l, k)),
        )?;
        hp.scales_agree(split)?;
        Ok(hp)
    }

    /// The largest top-k of the MoE layers.
    #[must_use]
    pub fn top_k_max(&self) -> usize {
        self.experts
            .iter()
            .flatten()
            .map(|e| e.top_k)
            .max()
            .unwrap_or(0)
    }

    /// The attention widths, as the `nemotron_h` reader's pieces take them.
    pub(super) fn attn(&self) -> Attn {
        Attn {
            n_head: self.n_head,
            key_len: self.key_len,
            value_len: self.value_len,
        }
    }

    /// The dims the keys give layer `l` of `kind`, ggml order.
    fn want(&self, l: usize, kind: Kind) -> Vec<(&'static str, Vec<u64>)> {
        let e = self.n_embd as u64;
        let mut want = vec![(ATTN_NORM, vec![e])];
        match kind {
            Kind::Mamba => {
                want.extend(mamba_dims(&self.ssm, e));
                want.extend([(SSM_IN_SCALE, vec![1]), (SSM_OUT_SCALE, vec![1])]);
            }
            Kind::Attention => want.extend(self.attn().dims(self.kv[l], e)),
            Kind::Moe => {
                let x = self.experts[l].expect("a MoE layer's experts are read");
                let (n, w) = (x.n_expert as u64, x.ff as u64);
                let (lat, sh) = (self.latent as u64, self.shared_ff as u64);
                want.extend([
                    (FFN_GATE_INP, vec![e, n]),
                    (EXP_PROBS_B, vec![n]),
                    (FFN_LATENT_DOWN, vec![e, lat]),
                    (FFN_LATENT_UP, vec![lat, e]),
                    (FFN_UP_EXPS, vec![lat, w, n]),
                    (FFN_DOWN_EXPS, vec![w, lat, n]),
                    (FFN_UP_SHEXP, vec![e, sh]),
                    (FFN_DOWN_SHEXP, vec![sh, e]),
                ]);
            }
        }
        want
    }

    /// A projection's per-tensor scale is one f32: the graph multiplies the
    /// projection's output by it (llama-graph.cpp:1520-1521).
    fn scales_agree(&self, split: &Split) -> Result<(), PlacementError> {
        for (l, _) in self
            .kinds
            .iter()
            .enumerate()
            .filter(|(_, k)| **k == Kind::Mamba)
        {
            for stem in [SSM_IN_SCALE, SSM_OUT_SCALE] {
                let name = format!("blk.{l}.{stem}");
                if let Some((_, t)) = split.find(&name)
                    && t.ty != GgmlType::F32
                {
                    return Err(PlacementError::Tensor {
                        name,
                        detail: format!("is {}, not the f32 scale the graph multiplies by", t.ty),
                    });
                }
            }
        }
        Ok(())
    }
}

/// Layer `l`'s entries of the two routed arrays agree with its kind: both
/// above 0 on a MoE layer, both 0 on any other. Each refusal names its key
/// and the layer.
fn routed_arrays_agree(
    split: &Split,
    l: usize,
    kind: Kind,
    ff: usize,
    top_k: usize,
) -> Result<(), PlacementError> {
    for (key, v) in [
        ("expert_feed_forward_length", ff),
        ("expert_used_count", top_k),
    ] {
        let moe = kind == Kind::Moe;
        if moe && v == 0 {
            return Err(metadata(
                split,
                key,
                format!("is 0 at layer {l}, {}", kind.what()),
            ));
        }
        if !moe && v != 0 {
            return Err(metadata(
                split,
                key,
                format!("is {v} at layer {l}, {}, which routes nothing", kind.what()),
            ));
        }
    }
    Ok(())
}

/// MoE layer `l`'s experts: the router's rows as the layer's expert count,
/// at most `expert_count` (`n_expert`), and a top-k of at most that count.
/// The stacks and the bias are held to the count by the dims check.
fn layer_experts(
    split: &Split,
    l: usize,
    n_expert: usize,
    ff: usize,
    top_k: usize,
) -> Result<Experts, PlacementError> {
    let name = names::ffn_gate_inp(l);
    let (_, router) = split.find(&name).ok_or_else(|| PlacementError::Tensor {
        name: name.clone(),
        detail: "is not in the file".to_string(),
    })?;
    let rows = match router.dims.as_slice() {
        [_, rows] => usize::try_from(*rows).ok(),
        _ => None,
    };
    let Some(rows) = rows.filter(|&r| r > 0) else {
        return Err(PlacementError::Tensor {
            name,
            detail: format!(
                "has dims {:?}; a router is a matrix of one row per expert",
                router.dims
            ),
        });
    };
    if rows > n_expert {
        return Err(PlacementError::Tensor {
            name,
            detail: format!("has {rows} rows, past expert_count {n_expert}"),
        });
    }
    if top_k > rows {
        return Err(metadata(
            split,
            "expert_used_count",
            format!("is {top_k} at layer {l}, past the layer's {rows} experts"),
        ));
    }
    Ok(Experts {
        n_expert: rows,
        top_k,
        ff,
    })
}
