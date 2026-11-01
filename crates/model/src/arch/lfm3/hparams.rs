//! Every `lfm3.*` key the description reads, and each layer's kind. A layer is a short conv where
//! `attention.head_count_kv` is 0 and an attention layer otherwise; an attention layer is sliding
//! where `attention.sliding_window_pattern` is true and full where it is false, a short-conv
//! layer's entry being false. The layer's marker tensor has to say the same thing. A key the
//! loader requires is required here. The rope turns the whole of `rope.dimension_count` values
//! of a head; the file's `rope_freqs.weight` holds one factor for each pair of them.

use gguf::Split;

use super::names::{
    ATTN_GATE, ATTN_K, ATTN_K_NORM, ATTN_NORM, ATTN_OUTPUT, ATTN_Q, ATTN_Q_NORM, ATTN_V, CONV_IN,
    CONV_OUT, CONV_TAPS, FFN_DOWN, FFN_GATE, FFN_NORM, FFN_UP, POST_ATTN_NORM, POST_FFN_NORM,
    ROPE_FREQS,
};
use super::roles;
use crate::arch::classify::rope_factors;
use crate::arch::{
    check_kv, dims_agree, finite_positive, metadata, mul, n_vocab, per_layer_bool, per_layer_count,
    plain_rope, positive,
};
use crate::placement::PlacementError;

/// What a layer is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// No key heads: a gated short conv.
    Conv,
    /// Attention over the last `attention.sliding_window` positions.
    Sliding,
    /// Attention over every position.
    Full,
}

/// The hyperparameters of one `lfm3` file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// `block_count`. This reader runs no next-token layer, so every block is a trunk layer.
    pub n_layer: usize,
    /// `embedding_length`: the residual width, and the short conv's channels.
    pub n_embd: usize,
    /// `context_length`.
    pub n_ctx: usize,
    /// `vocab_size`, else the token list's length.
    pub n_vocab: usize,
    /// `attention.layer_norm_rms_epsilon`: every norm's, the sandwich norms' too.
    pub rms_eps: f32,
    /// `attention.head_count`.
    pub n_head: usize,
    /// `attention.key_length`.
    pub key_len: usize,
    /// `attention.value_length`.
    pub value_len: usize,
    /// `attention.head_count_kv`, shared or per layer, resolved per layer. 0 on a short-conv
    /// layer.
    pub kv: Vec<usize>,
    /// `feed_forward_length`, shared or per layer, resolved per layer.
    pub ff: Vec<usize>,
    /// Every layer's kind, in order.
    pub kinds: Vec<Kind>,
    /// `attention.sliding_window`: the positions a sliding layer attends, its own among them.
    pub window: usize,
    /// `shortconv.l_cache`: the taps of a short conv's filter.
    pub taps: usize,
    /// `rope.dimension_count`: the values of a head the rope turns, in pairs `(i, i + dims / 2)`.
    pub rope_dims: usize,
    /// `rope.freq_base`: θ of the full layers.
    pub rope_base: f32,
    /// `rope.freq_base_swa`: θ of the sliding layers.
    pub rope_base_swa: f32,
    /// Whether the file carries `rope_freqs.weight`, which the full layers' rope divides its pair
    /// frequencies by. The sliding layers' rope has no factors.
    pub rope_factors: bool,
}

impl Hparams {
    /// Every value from `split`'s headers. A missing required key, a per-layer array of the
    /// wrong length, a layer whose tensors disagree with the arrays, a tensor of other dims than
    /// the keys give, or a rope factor table that is not one value a pair is an error naming the
    /// key or the tensor.
    pub fn read(split: &Split) -> Result<Hparams, PlacementError> {
        let n_layer = positive(split, "block_count")?;
        let n_embd = positive(split, "embedding_length")?;
        let n_ctx = positive(split, "context_length")?;
        let rms_eps = finite_positive(split, "attention.layer_norm_rms_epsilon")?;
        let n_head = positive(split, "attention.head_count")?;
        let key_len = positive(split, "attention.key_length")?;
        let value_len = positive(split, "attention.value_length")?;
        let n_vocab = n_vocab(split)?;
        let kv = per_layer_count(split, "attention.head_count_kv", n_layer)?;
        let ff = per_layer_count(split, "feed_forward_length", n_layer)?;
        if let Some(l) = ff.iter().position(|&f| f == 0) {
            return Err(metadata(
                split,
                "feed_forward_length",
                format!("is 0 at layer {l}; every layer carries a feed-forward block"),
            ));
        }
        let taps = positive(split, "shortconv.l_cache")?;
        let window = positive(split, "attention.sliding_window")?;
        let pattern = per_layer_bool(split, "attention.sliding_window_pattern", n_layer)?;
        let rope_dims = positive(split, "rope.dimension_count")?;
        if !rope_dims.is_multiple_of(2) || rope_dims > key_len {
            return Err(metadata(
                split,
                "rope.dimension_count",
                format!("is {rope_dims}, for a key head of {key_len}"),
            ));
        }
        plain_rope(split)?;
        let rope_base = finite_positive(split, "rope.freq_base")?;
        let rope_base_swa = finite_positive(split, "rope.freq_base_swa")?;
        mul(split, "attention.key_length", n_head, key_len)?;
        mul(split, "attention.value_length", n_head, value_len)?;
        let kinds = layer_kinds(split, &kv, &pattern)?;
        for (l, (&heads, &kind)) in kv.iter().zip(&kinds).enumerate() {
            if kind != Kind::Conv {
                check_kv(split, n_head, l, heads)?;
            }
        }
        let rope_factors = rope_factors(split, rope_dims)?;
        let hp = Hparams {
            n_layer,
            n_embd,
            n_ctx,
            n_vocab,
            rms_eps,
            n_head,
            key_len,
            value_len,
            kv,
            ff,
            kinds,
            window,
            taps,
            rope_dims,
            rope_base,
            rope_base_swa,
            rope_factors,
        };
        roles::require(split, &hp.kinds)?;
        dims_agree(
            split,
            hp.n_embd,
            hp.n_vocab,
            hp.kinds.iter().enumerate().map(|(l, &k)| hp.want(l, k)),
        )?;
        Ok(hp)
    }

    /// The values of the file's `rope_freqs.weight` for the full layers' rope: one f32 a pair of
    /// [`Hparams::rope_dims`], in the file's order, or `None` when the file carries no table. The
    /// header read checked the table's dims and type; this reads its bytes, so a split without its
    /// data is refused by name. The rope divides each pair's angle by its value, and
    /// `chain::rope::RopeSpec::with_factors` refuses a value it cannot divide by.
    pub fn rope_factor_values(&self, split: &Split) -> Result<Option<Vec<f32>>, PlacementError> {
        if !self.rope_factors {
            return Ok(None);
        }
        let tensor = |detail: String| PlacementError::Tensor {
            name: ROPE_FREQS.to_string(),
            detail,
        };
        let bytes = split
            .data(ROPE_FREQS)
            .map_err(|e| tensor(format!("cannot be read: {e}")))?;
        factors_from_le(bytes, self.rope_dims / 2)
            .map(Some)
            .map_err(tensor)
    }

    /// The dims the keys give layer `l` of `kind`, ggml order.
    fn want(&self, l: usize, kind: Kind) -> Vec<(&'static str, Vec<u64>)> {
        let e = self.n_embd as u64;
        let ff = self.ff[l] as u64;
        let mut want = vec![
            (ATTN_NORM, vec![e]),
            (POST_ATTN_NORM, vec![e]),
            (FFN_NORM, vec![e]),
            (POST_FFN_NORM, vec![e]),
            (FFN_GATE, vec![e, ff]),
            (FFN_UP, vec![e, ff]),
            (FFN_DOWN, vec![ff, e]),
        ];
        match kind {
            Kind::Conv => want.extend([
                (CONV_IN, vec![e, 3 * e]),
                (CONV_TAPS, vec![self.taps as u64, e]),
                (CONV_OUT, vec![e, e]),
            ]),
            Kind::Sliding | Kind::Full => {
                let (heads, kv) = (self.n_head as u64, self.kv[l] as u64);
                let (k, v) = (self.key_len as u64, self.value_len as u64);
                want.extend([
                    (ATTN_Q, vec![e, heads * k]),
                    (ATTN_K, vec![e, kv * k]),
                    (ATTN_V, vec![e, kv * v]),
                    (ATTN_OUTPUT, vec![heads * v, e]),
                    (ATTN_GATE, vec![e, heads]),
                    (ATTN_Q_NORM, vec![k]),
                    (ATTN_K_NORM, vec![k]),
                ]);
            }
        }
        want
    }
}

/// Every layer's kind from the head counts and the sliding pattern, as the loader decides it: no
/// key heads is a short conv, else the pattern's entry says sliding or full. Each layer carries
/// exactly one of the two marker tensors, the short conv's input projection and the query
/// projection, and the one it carries has to be the kind the arrays say; a pattern entry that
/// is true on a short-conv layer is refused.
fn layer_kinds(split: &Split, kv: &[usize], pattern: &[bool]) -> Result<Vec<Kind>, PlacementError> {
    kv.iter()
        .zip(pattern)
        .enumerate()
        .map(|(l, (&heads, &sliding))| {
            let present = |stem: &str| split.find(&format!("blk.{l}.{stem}")).is_some();
            let (conv, attn) = (present(CONV_IN), present(ATTN_Q));
            let kind = match (conv, attn) {
                (true, false) => Kind::Conv,
                (false, true) if sliding => Kind::Sliding,
                (false, true) => Kind::Full,
                _ => {
                    return Err(PlacementError::Tensor {
                        name: format!("blk.{l}.{CONV_IN}"),
                        detail: format!(
                            "layer {l} needs exactly one of {CONV_IN} and {ATTN_Q} ({conv} / {attn})"
                        ),
                    });
                }
            };
            if (heads == 0) != (kind == Kind::Conv) {
                return Err(metadata(
                    split,
                    "attention.head_count_kv",
                    format!("is {heads} at layer {l}, and its tensors make it {kind:?}"),
                ));
            }
            if sliding && kind == Kind::Conv {
                return Err(metadata(
                    split,
                    "attention.sliding_window_pattern",
                    format!("is true at layer {l}, which is a short-conv layer"),
                ));
            }
            Ok(kind)
        })
        .collect()
}

/// `pairs` little-endian f32 from `bytes`, which hold exactly that many.
fn factors_from_le(bytes: &[u8], pairs: usize) -> Result<Vec<f32>, String> {
    if bytes.len() != pairs * 4 {
        return Err(format!(
            "has {} bytes; {pairs} pairs of f32 are {}",
            bytes.len(),
            pairs * 4
        ));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}

#[cfg(test)]
mod tests {
    use gguf::Split;

    use super::super::names::ROPE_FREQS;
    use super::super::spec::tests::{keys, tensors};
    use super::{Hparams, factors_from_le};
    use crate::arch::synthetic::{V, header_shaped};

    /// The table's bytes are little-endian f32, one a pair, in the file's order: `1.0` is
    /// `00 00 80 3f`, and a table of another byte count is refused by name. Mutant: the decode is
    /// big-endian, or the count check is dropped.
    #[test]
    fn the_factor_bytes_are_little_endian_f32_a_pair() {
        let mut bytes = Vec::new();
        for f in [1.0f32, 1e30] {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        assert_eq!(bytes[..4], [0x00, 0x00, 0x80, 0x3f]);
        assert_eq!(factors_from_le(&bytes, 2), Ok(vec![1.0, 1e30]));
        let err = factors_from_le(&bytes, 3).expect_err("a short table");
        assert_eq!(err, "has 8 bytes; 3 pairs of f32 are 12");
    }

    /// A file with the table gives one value a pair of the rope's dims, a file without gives none.
    /// The synthetic file's tensor data are zeros, so this pins the count and the read path; the
    /// real file's values are `hw_lfm3_rope_factors`'s. Mutant: the table is read for a file that
    /// has none, or the count is the dims and not the pairs.
    #[test]
    fn a_file_with_the_table_gives_a_value_a_pair_and_one_without_gives_none() {
        let global = [("tokenizer.ggml.pre", V::Str("lfm2"))];
        let mut bare = tensors();
        bare.retain(|(n, _)| n != ROPE_FREQS);
        for (tag, file, want) in [
            ("lfm3-factor-values", tensors(), Some(vec![0.0; 4])),
            ("lfm3-factor-none", bare, None),
        ] {
            let path = header_shaped(tag, "lfm3", &keys(), &global, &file);
            let split = Split::open(&path).expect("the synthetic header opens");
            let hp = Hparams::read(&split).expect("the synthetic header reads");
            let got = hp.rope_factor_values(&split);
            let _ = std::fs::remove_file(&path);
            assert_eq!(got.expect("a readable table"), want, "{tag}");
        }
    }
}
