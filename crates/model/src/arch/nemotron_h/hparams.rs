//! Every `nemotron_h.*` key the description reads, and each layer's kind
//! from the per-layer arrays. A layer is Mamba-2 where both
//! `attention.head_count_kv` and `feed_forward_length` are 0, attention
//! where only the feed-forward length is 0, and a dense MLP otherwise
//! (nemotron-h.cpp, the recurrent test then the feed-forward test). The
//! layer's marker tensor has to say the same thing. A key the loader
//! requires is required here. The state width is 128: the scan to come is
//! written for that width.

use gguf::Split;

use super::names::{self, ATTN_NORM};
use super::roles;
use crate::arch::stems::{Stem, required};
use crate::arch::{
    meta_arr, meta_f32, meta_usize, metadata, n_vocab, optional_usize, positive,
};
use crate::placement::PlacementError;

/// What a layer is, from the two per-layer arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Both arrays 0.
    Mamba,
    /// Feed-forward length 0, KV heads not.
    Attention,
    /// A positive feed-forward length.
    Mlp,
}

/// The Mamba-2 widths, from the `ssm.*` keys and the products the projections
/// are built at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ssm {
    /// `ssm.conv_kernel`.
    pub d_conv: usize,
    /// `ssm.inner_size`.
    pub d_inner: usize,
    /// `ssm.state_size`. 128: a scan written for another width is refused.
    pub d_state: usize,
    /// `ssm.group_count`.
    pub n_groups: usize,
    /// `ssm.time_step_rank`: the scan's heads.
    pub n_heads: usize,
    /// `d_inner / n_heads`.
    pub head_dim: usize,
    /// Input-projection columns: `2 * d_inner + 2 * n_groups * d_state + n_heads`.
    pub d_in_proj: usize,
    /// Conv columns: `d_inner + 2 * n_groups * d_state`.
    pub conv_dim: usize,
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
    /// `rope.dimension_count`, read and not applied: an attention layer
    /// rotates nothing.
    pub rope_dims: usize,
    /// The Mamba-2 widths. One set for every Mamba-2 layer.
    pub ssm: Ssm,
    /// `attention.head_count_kv`, per layer. 0 on a Mamba-2 layer.
    pub kv: Vec<usize>,
    /// `feed_forward_length`, per layer. 0 on a Mamba-2 or attention layer.
    pub ff: Vec<usize>,
    /// Every layer's kind, in order.
    pub kinds: Vec<Kind>,
    /// The keys this read took a default for. This reader takes none.
    pub defaults: Vec<String>,
}

impl Hparams {
    /// Every value from `split`'s headers. A missing required key, a state
    /// width other than 128, a per-layer array of the wrong length, or a
    /// layer whose tensors disagree with the arrays is an error naming the
    /// key or the tensor.
    pub fn read(split: &Split) -> Result<Hparams, PlacementError> {
        let n_layer = positive(split, "block_count")?;
        no_nextn(split)?;
        let n_embd = positive(split, "embedding_length")?;
        let n_ctx = positive(split, "context_length")?;
        let n_head = positive(split, "attention.head_count")?;
        let key_len = positive(split, "attention.key_length")?;
        let value_len = positive(split, "attention.value_length")?;
        let rms_eps = meta_f32(split, "attention.layer_norm_rms_epsilon")?;
        if !rms_eps.is_finite() || rms_eps <= 0.0 {
            return Err(metadata(
                split,
                "attention.layer_norm_rms_epsilon",
                format!("is {rms_eps}, not a finite positive epsilon"),
            ));
        }
        // Loaded, and not a reason to refuse the file: the token forward
        // does not rotate. A missing key is still named.
        let rope_dims = meta_usize(split, "rope.dimension_count")?;
        if rope_dims == 0 || !rope_dims.is_multiple_of(2) || rope_dims > key_len {
            return Err(metadata(
                split,
                "rope.dimension_count",
                format!("is {rope_dims}, for a key head of {key_len}"),
            ));
        }
        let ssm = ssm(split)?;
        let n_vocab = n_vocab(split)?;
        let kv = per_layer_count(split, "attention.head_count_kv", n_layer)?;
        let ff = per_layer_count(split, "feed_forward_length", n_layer)?;
        let kinds = kinds(split, &kv, &ff)?;
        for (l, (&heads, &kind)) in kv.iter().zip(&kinds).enumerate() {
            if kind == Kind::Attention && (heads == 0 || !n_head.is_multiple_of(heads)) {
                return Err(metadata(
                    split,
                    "attention.head_count_kv",
                    format!("is {heads} at layer {l} for {n_head} query heads"),
                ));
            }
        }
        let hp = Hparams {
            n_layer,
            n_embd,
            n_ctx,
            n_vocab,
            rms_eps,
            n_head,
            key_len,
            value_len,
            rope_dims,
            ssm,
            kv,
            ff,
            kinds,
            defaults: Vec::new(),
        };
        hp.tensors_agree(split)?;
        hp.dims_agree(split)?;
        Ok(hp)
    }

    /// Every layer carries its kind's required tensors and nothing of the
    /// other kinds' except `attn_norm`, which every layer carries. The
    /// model-level three are required. The first missing or misplaced name,
    /// with how many more there are.
    fn tensors_agree(&self, split: &Split) -> Result<(), PlacementError> {
        let has = |name: &str| split.find(name).is_some();
        let mut missing = Vec::new();
        let mut misplaced = Vec::new();
        for name in [names::token_embd(), names::output_norm(), names::output()] {
            if !has(&name) {
                missing.push(name);
            }
        }
        for (l, &kind) in self.kinds.iter().enumerate() {
            let layer_has = |stem: &str| has(&format!("blk.{l}.{stem}"));
            missing.extend(
                required(roles::stems_of(kind))
                    .filter(|s| !layer_has(s))
                    .map(|s| format!("blk.{l}.{s}")),
            );
            let banned = banned(kind);
            misplaced.extend(
                banned
                    .iter()
                    .filter(|s| layer_has(s))
                    .map(|s| format!("blk.{l}.{s}")),
            );
        }
        if let Some(name) = missing.first() {
            return Err(PlacementError::Tensor {
                name: name.clone(),
                detail: format!(
                    "is not in the file ({} tensor(s) the layers need are missing: {})",
                    missing.len(),
                    missing.join(", ")
                ),
            });
        }
        if let Some(name) = misplaced.first() {
            return Err(PlacementError::Tensor {
                name: name.clone(),
                detail: format!(
                    "is in the file, and its layer's kind carries none ({} such: {})",
                    misplaced.len(),
                    misplaced.join(", ")
                ),
            });
        }
        Ok(())
    }

    /// The tensors whose dims the keys name hold exactly those dims. A
    /// tensor the file does not carry is skipped: [`Self::tensors_agree`]
    /// already named it. The first tensor of other dims, by name.
    fn dims_agree(&self, split: &Split) -> Result<(), PlacementError> {
        let mut off = Vec::new();
        let check = |off: &mut Vec<(String, String)>, name: String, dims: Vec<u64>| {
            if let Some((_, info)) = split.find(&name)
                && info.dims != dims
            {
                off.push((name, format!("{:?} (want {dims:?})", info.dims)));
            }
        };
        let e = self.n_embd as u64;
        let vocab = self.n_vocab as u64;
        check(&mut off, names::token_embd(), vec![e, vocab]);
        check(&mut off, names::output(), vec![e, vocab]);
        check(&mut off, names::output_norm(), vec![e]);
        for (l, &kind) in self.kinds.iter().enumerate() {
            for (stem, dims) in self.want(l, kind) {
                check(&mut off, format!("blk.{l}.{stem}"), dims);
            }
        }
        match off.first() {
            None => Ok(()),
            Some((name, _)) => Err(PlacementError::Tensor {
                name: name.clone(),
                detail: format!(
                    "has other dims than the keys give the tensor ({} such: {})",
                    off.len(),
                    off.iter()
                        .map(|(n, d)| format!("{n} {d}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            }),
        }
    }

    /// The dims the keys give layer `l` of `kind`, ggml order.
    fn want(&self, l: usize, kind: Kind) -> Vec<(&'static str, Vec<u64>)> {
        let e = self.n_embd as u64;
        let norm = (ATTN_NORM, vec![e]);
        match kind {
            Kind::Mamba => {
                let s = &self.ssm;
                let (heads, groups) = (s.n_heads as u64, s.n_groups as u64);
                vec![
                    norm,
                    (names::SSM_A, vec![1, heads]),
                    (names::SSM_D, vec![1, heads]),
                    (names::SSM_DT_BIAS, vec![heads]),
                    (names::SSM_CONV, vec![s.d_conv as u64, s.conv_dim as u64]),
                    (names::SSM_CONV_BIAS, vec![s.conv_dim as u64]),
                    (names::SSM_IN, vec![e, s.d_in_proj as u64]),
                    (
                        names::SSM_NORM,
                        vec![(s.d_inner / s.n_groups) as u64, groups],
                    ),
                    (names::SSM_OUT, vec![s.d_inner as u64, e]),
                ]
            }
            Kind::Attention => {
                let (h, kv) = (self.n_head as u64, self.kv[l] as u64);
                let (k, v) = (self.key_len as u64, self.value_len as u64);
                vec![
                    norm,
                    (names::ATTN_Q, vec![e, h * k]),
                    (names::ATTN_K, vec![e, kv * k]),
                    (names::ATTN_V, vec![e, kv * v]),
                    (names::ATTN_OUTPUT, vec![h * v, e]),
                ]
            }
            Kind::Mlp => {
                let ff = self.ff[l] as u64;
                vec![
                    norm,
                    (names::FFN_UP, vec![e, ff]),
                    (names::FFN_DOWN, vec![ff, e]),
                ]
            }
        }
    }
}

/// `ssm.*`. The state width is checked before any tensor is walked.
fn ssm(split: &Split) -> Result<Ssm, PlacementError> {
    let d_conv = positive(split, "ssm.conv_kernel")?;
    let d_inner = positive(split, "ssm.inner_size")?;
    let d_state = meta_usize(split, "ssm.state_size")?;
    if d_state != 128 {
        return Err(metadata(
            split,
            "ssm.state_size",
            format!("is {d_state}, not 128"),
        ));
    }
    let n_groups = positive(split, "ssm.group_count")?;
    let n_heads = positive(split, "ssm.time_step_rank")?;
    if !d_inner.is_multiple_of(n_heads) {
        return Err(metadata(
            split,
            "ssm.inner_size",
            format!("is {d_inner}, not divisible by ssm.time_step_rank {n_heads}"),
        ));
    }
    if !d_inner.is_multiple_of(n_groups) {
        return Err(metadata(
            split,
            "ssm.inner_size",
            format!("is {d_inner}, not divisible by ssm.group_count {n_groups}"),
        ));
    }
    let groups_state = mul(split, "ssm.group_count", n_groups, d_state)?;
    let twice_groups = mul(split, "ssm.group_count", 2, groups_state)?;
    let twice_inner = mul(split, "ssm.inner_size", 2, d_inner)?;
    let conv_dim = add(split, "ssm.inner_size", d_inner, twice_groups)?;
    let d_in_proj = add(
        split,
        "ssm.inner_size",
        add(split, "ssm.inner_size", twice_inner, twice_groups)?,
        n_heads,
    )?;
    Ok(Ssm {
        d_conv,
        d_inner,
        d_state,
        n_groups,
        n_heads,
        head_dim: d_inner / n_heads,
        d_in_proj,
        conv_dim,
    })
}

/// `a * b`, or the error that names `suffix`.
fn mul(split: &Split, suffix: &str, a: usize, b: usize) -> Result<usize, PlacementError> {
    a.checked_mul(b)
        .ok_or_else(|| metadata(split, suffix, format!("{a} * {b} does not fit")))
}

/// `a + b`, or the error that names `suffix`.
fn add(split: &Split, suffix: &str, a: usize, b: usize) -> Result<usize, PlacementError> {
    a.checked_add(b)
        .ok_or_else(|| metadata(split, suffix, format!("{a} + {b} does not fit")))
}

/// `nextn_predict_layers`: absent or 0. A count is refused by name. This
/// reader runs no next-token layer.
fn no_nextn(split: &Split) -> Result<(), PlacementError> {
    let key = "nextn_predict_layers";
    let Some(n) = optional_usize(split, key)? else {
        return Ok(());
    };
    if n != 0 {
        return Err(metadata(
            split,
            key,
            format!("is {n}; this reader runs no next-token layer"),
        ));
    }
    Ok(())
}

/// `<architecture>.<suffix>` as a per-layer array of counts. Zero is a
/// count: it is how a layer says it has no KV heads, or no feed-forward
/// block. A scalar is refused by name.
fn per_layer_count(
    split: &Split,
    suffix: &str,
    n_layer: usize,
) -> Result<Vec<usize>, PlacementError> {
    let items = meta_arr(split, suffix)?;
    if items.len() != n_layer {
        return Err(metadata(
            split,
            suffix,
            format!("has {} values for {n_layer} layers", items.len()),
        ));
    }
    items
        .iter()
        .enumerate()
        .map(|(l, v)| {
            v.as_unsigned()
                .and_then(|x| usize::try_from(x).ok())
                .ok_or_else(|| metadata(split, suffix, format!("has no count at layer {l}")))
        })
        .collect()
}

/// Both arrays 0 is Mamba-2; else a zero feed-forward length is attention;
/// else the dense MLP.
fn kind_of(kv: usize, ff: usize) -> Kind {
    if kv == 0 && ff == 0 {
        Kind::Mamba
    } else if ff == 0 {
        Kind::Attention
    } else {
        Kind::Mlp
    }
}

/// Every layer's kind from the two arrays, the marker tensor agreeing:
/// `ssm_in` Mamba-2, `attn_q` attention, `ffn_up` the dense MLP. Exactly
/// one of the three.
fn kinds(split: &Split, kv: &[usize], ff: &[usize]) -> Result<Vec<Kind>, PlacementError> {
    kv.iter()
        .zip(ff)
        .enumerate()
        .map(|(l, (&kv, &ff))| {
            let said = kind_of(kv, ff);
            let present = |stem: &str| split.find(&format!("blk.{l}.{stem}")).is_some();
            let (ssm, attn, mlp) = (
                present(names::SSM_IN),
                present(names::ATTN_Q),
                present(names::FFN_UP),
            );
            let kind = match (ssm, attn, mlp) {
                (true, false, false) => Kind::Mamba,
                (false, true, false) => Kind::Attention,
                (false, false, true) => Kind::Mlp,
                _ => {
                    return Err(PlacementError::Tensor {
                        name: format!("blk.{l}.{}", names::SSM_IN),
                        detail: format!(
                            "layer {l} needs exactly one of {}, {} and {} ({ssm} / {attn} / {mlp})",
                            names::SSM_IN,
                            names::ATTN_Q,
                            names::FFN_UP
                        ),
                    });
                }
            };
            if kind != said {
                return Err(metadata(
                    split,
                    "attention.head_count_kv",
                    format!(
                        "with feed_forward_length makes layer {l} {said:?}, and its tensors make \
                         it {kind:?}"
                    ),
                ));
            }
            Ok(kind)
        })
        .collect()
}

/// The other kinds' stems, `attn_norm` aside: every layer carries that one.
fn banned(kind: Kind) -> Vec<&'static str> {
    let groups: [&[Stem]; 2] = match kind {
        Kind::Mamba => [roles::ATTENTION, roles::MLP],
        Kind::Attention => [roles::MAMBA, roles::MLP],
        Kind::Mlp => [roles::MAMBA, roles::ATTENTION],
    };
    groups
        .into_iter()
        .flat_map(|g| g.iter())
        .map(|s| s.name)
        .filter(|s| *s != ATTN_NORM)
        .collect()
}
