//! Every `nemotron_h.*` key the description reads, and each layer's kind
//! from the shared or per-layer counts. A layer is Mamba-2 where both
//! `attention.head_count_kv` and `feed_forward_length` are 0, attention
//! where only the feed-forward length is 0, and a dense MLP otherwise
//! (nemotron-h.cpp, the recurrent test then the feed-forward test). The
//! layer's marker tensor has to say the same thing. A key the loader
//! requires is required here. The state width is 128: the scan to come is
//! written for that width. The Mamba-2 and attention layers, the kinds and
//! the tensors each kind carries are read by the hybrid trunk's common
//! pieces (`arch::hybrid`).

use gguf::Split;

use super::names::{self, ATTN_NORM};
use super::roles;
pub use crate::arch::hybrid::Ssm;
use crate::arch::hybrid::{Attn, attn, layer_kinds, layers_agree, mamba_dims, no_nextn, ssm};
use crate::arch::{finite_positive, n_vocab, per_layer_count, positive};
use crate::placement::PlacementError;

/// What a layer is, from the two resolved per-layer counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Both counts 0.
    Mamba,
    /// Feed-forward length 0, KV heads not.
    Attention,
    /// A positive feed-forward length.
    Mlp,
}

/// The hyperparameters of one `nemotron_h` file.
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
    /// `attention.head_count_kv`, shared or per layer, resolved per layer.
    /// 0 on a Mamba-2 layer.
    pub kv: Vec<usize>,
    /// `feed_forward_length`, shared or per layer, resolved per layer.
    /// 0 on a Mamba-2 or attention layer.
    pub ff: Vec<usize>,
    /// Every layer's kind, in order.
    pub kinds: Vec<Kind>,
}

impl Hparams {
    /// Every value from `split`'s headers. A missing required key, a state
    /// width other than 128, a per-layer array of the wrong length, or a
    /// layer whose tensors disagree with the counts is an error naming the
    /// key or the tensor.
    pub fn read(split: &Split) -> Result<Hparams, PlacementError> {
        let n_layer = positive(split, "block_count")?;
        no_nextn(split)?;
        let n_embd = positive(split, "embedding_length")?;
        let n_ctx = positive(split, "context_length")?;
        let rms_eps = finite_positive(split, "attention.layer_norm_rms_epsilon")?;
        let attn = attn(split)?;
        let ssm = ssm(split)?;
        let n_vocab = n_vocab(split)?;
        let kv = per_layer_count(split, "attention.head_count_kv", n_layer)?;
        let ff = per_layer_count(split, "feed_forward_length", n_layer)?;
        let kinds = layer_kinds(
            split,
            &kv,
            &ff,
            [
                (names::SSM_IN, Kind::Mamba),
                (names::ATTN_Q, Kind::Attention),
                (names::FFN_UP, Kind::Mlp),
            ],
        )?;
        for (l, (&heads, &kind)) in kv.iter().zip(&kinds).enumerate() {
            if kind == Kind::Attention {
                attn.check_layer(split, l, heads)?;
            }
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
            ff,
            kinds,
        };
        layers_agree(
            split,
            &hp.kinds,
            |kind| vec![roles::stems_of(kind)],
            &[Kind::Mamba, Kind::Attention, Kind::Mlp],
        )?;
        crate::arch::dims_agree(
            split,
            hp.n_embd,
            hp.n_vocab,
            hp.kinds.iter().enumerate().map(|(l, &k)| hp.want(l, k)),
        )?;
        Ok(hp)
    }

    /// The dims the keys give layer `l` of `kind`, ggml order.
    fn want(&self, l: usize, kind: Kind) -> Vec<(&'static str, Vec<u64>)> {
        let e = self.n_embd as u64;
        let mut want = vec![(ATTN_NORM, vec![e])];
        match kind {
            Kind::Mamba => want.extend(mamba_dims(&self.ssm, e)),
            Kind::Attention => want.extend(
                Attn {
                    n_head: self.n_head,
                    key_len: self.key_len,
                    value_len: self.value_len,
                }
                .dims(self.kv[l], e),
            ),
            Kind::Mlp => {
                let ff = self.ff[l] as u64;
                want.extend([(names::FFN_UP, vec![e, ff]), (names::FFN_DOWN, vec![ff, e])]);
            }
        }
        want
    }
}
