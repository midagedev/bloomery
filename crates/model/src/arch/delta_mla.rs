//! The trunk `raon1` and `kimi_k3` share: every layer mixes with a delta rule or with a gated
//! latent attention (MLA), told apart by `attention.head_count_kv` (0 and 1), the first layers are
//! dense `situ` blocks and the rest routed. What the two files hold the same lives here: the
//! latent layer's widths with the keys that fix them, the tensors those widths give (stems and
//! dims) and the mixer description; the `situ` activation's keys; the dense lead; and each
//! layer's kind from the per-layer array. What stays with a family: its delta rule's own gates
//! and dims, the residual's form (hyper-connections, the cross-layer mix), the keys only it
//! reads, and the plan that places the layers.

use gguf::Split;
use models::{Act, Latent, LatentOut, LatentUp, OutGate, Source};

use crate::arch::stems::{Stem, req};
use crate::arch::{finite_positive, meta_usize, metadata, per_layer_count, positive, spec_u32};
use crate::placement::{PlacementError, Role};

/// A latent layer's mixer: the query and key-value latents with their norms, the absorbed
/// per-head up projections, the output gate and the output projection.
pub(in crate::arch) const MLA: &[Stem] = &[
    req("attn_norm.weight", Role::Attention),
    req("attn_q_a.weight", Role::Attention),
    req("attn_q_a_norm.weight", Role::Attention),
    req("attn_q_b.weight", Role::Attention),
    req("attn_kv_a_mqa.weight", Role::Attention),
    req("attn_kv_a_norm.weight", Role::Attention),
    req("attn_k_b.weight", Role::Attention),
    req("attn_v_b.weight", Role::Attention),
    req("attn_gate.weight", Role::Attention),
    req("attn_output.weight", Role::Attention),
];

/// The latent layers' widths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mla {
    /// `attention.q_lora_rank`: the query latent's values.
    pub q_lora: usize,
    /// `attention.kv_lora_rank`: the key and value latent's values, the width of the value the
    /// heads attend over before their up projection.
    pub kv_lora: usize,
    /// `attention.key_length_mla`: a query head's values: the no-position values then the
    /// unrotated ones.
    pub head_k: usize,
    /// `attention.value_length_mla`: a head's output values.
    pub head_v: usize,
    /// `rope.dimension_count`: the key values beside the latent that no rope turns, shared by
    /// every head.
    pub pe: usize,
}

impl Mla {
    /// A query head's values before the absorption: `head_k` less the unrotated ones.
    #[must_use]
    pub fn nope(&self) -> usize {
        self.head_k - self.pe
    }

    /// The cached row's values a position: the latent and the unrotated key values
    /// (`attention.key_length`). The value is the latent alone.
    #[must_use]
    pub fn row(&self) -> usize {
        self.kv_lora + self.pe
    }

    /// The widths of `split`'s keys; a key length that leaves a head no value besides the
    /// unrotated ones, and a generic key length other than the cached row's, are errors naming the
    /// key. The generic value length is not read here: a family states what its file holds there.
    pub(in crate::arch) fn read(split: &Split) -> Result<Mla, PlacementError> {
        let mla = Mla {
            q_lora: positive(split, "attention.q_lora_rank")?,
            kv_lora: positive(split, "attention.kv_lora_rank")?,
            head_k: positive(split, "attention.key_length_mla")?,
            head_v: positive(split, "attention.value_length_mla")?,
            pe: positive(split, "rope.dimension_count")?,
        };
        if mla.pe >= mla.head_k {
            return Err(metadata(
                split,
                "attention.key_length_mla",
                format!(
                    "is {}, not above rope.dimension_count {}: a head needs values besides the \
                     unrotated ones",
                    mla.head_k, mla.pe
                ),
            ));
        }
        // The cache holds one row of the latent and its unrotated values as the key, so the
        // generic key length is the row's width.
        let got = positive(split, "attention.key_length")?;
        if got != mla.row() {
            return Err(metadata(
                split,
                "attention.key_length",
                format!(
                    "is {got}, not kv_lora_rank + rope.dimension_count ({}): the cached row is \
                     the latent and its unrotated values",
                    mla.row()
                ),
            ));
        }
        Ok(mla)
    }

    /// The tensors of a latent layer by stem and the dims llama.cpp creates them with on an
    /// `e`-value row of `h` heads, ggml order: [`MLA`]'s.
    #[must_use]
    pub(in crate::arch) fn dims(&self, e: u64, h: u64) -> [(&'static str, Vec<u64>); 10] {
        let (q, kv, k, v) = (
            self.q_lora as u64,
            self.kv_lora as u64,
            self.head_k as u64,
            self.head_v as u64,
        );
        [
            ("attn_norm.weight", vec![e]),
            ("attn_q_a.weight", vec![e, q]),
            ("attn_q_a_norm.weight", vec![q]),
            ("attn_q_b.weight", vec![q, h * k]),
            ("attn_kv_a_mqa.weight", vec![e, self.row() as u64]),
            ("attn_kv_a_norm.weight", vec![kv]),
            ("attn_k_b.weight", vec![self.nope() as u64, kv, h]),
            ("attn_v_b.weight", vec![kv, v, h]),
            ("attn_gate.weight", vec![e, h * v]),
            ("attn_output.weight", vec![h * v, e]),
        ]
    }

    /// The layer's mixer description over `n_head` query heads: it caches the latent and the
    /// unrotated key values no rope turns a position, absorbs the per-head up projections and
    /// gates its heads' outputs by a projection of the layer input.
    pub(in crate::arch) fn latent(&self, n_head: usize) -> Result<Latent, PlacementError> {
        Ok(Latent {
            heads: spec_u32("attention.head_count", n_head)?,
            q_lora: spec_u32("attention.q_lora_rank", self.q_lora)?,
            latent: spec_u32("attention.kv_lora_rank", self.kv_lora)?,
            up: LatentUp::Absorbed {
                qk: spec_u32("attention.key_length_mla", self.head_k)?,
                v: spec_u32("attention.value_length_mla", self.head_v)?,
            },
            rope: None,
            q_head_norm: false,
            ssmax: false,
            pe: spec_u32("rope.dimension_count", self.pe)?,
            cache: Source::Own(()),
            out_gate: OutGate::Proj,
            out: LatentOut::Plain,
            window: None,
            sinks: false,
            compress: None,
            select: None,
        })
    }
}

/// The `situ` activation's keys: the gate's and the up path's bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Situ {
    /// `activation.situ_beta`: the gate's bound.
    pub gate_beta: f32,
    /// `activation.situ_linear_beta`: the up path's bound.
    pub value_beta: f32,
}

impl Situ {
    /// The two bounds of `split`'s keys, each finite and above 0.
    pub(in crate::arch) fn read(split: &Split) -> Result<Situ, PlacementError> {
        Ok(Situ {
            gate_beta: finite_positive(split, "activation.situ_beta")?,
            value_beta: finite_positive(split, "activation.situ_linear_beta")?,
        })
    }

    /// The activation the dense layers and the experts take.
    #[must_use]
    pub(in crate::arch) fn act(&self) -> Act {
        Act::Situ {
            gate_beta: self.gate_beta,
            value_beta: self.value_beta,
        }
    }
}

/// `leading_dense_block_count`, required, below the layer count: a trunk with no routed layer
/// is another reader's decision.
pub(in crate::arch) fn dense_lead_of(
    split: &Split,
    n_layer: usize,
) -> Result<usize, PlacementError> {
    let key = "leading_dense_block_count";
    let lead = meta_usize(split, key)?;
    if lead >= n_layer {
        return Err(metadata(
            split,
            key,
            format!("is {lead} for {n_layer} layers; at least one layer must route"),
        ));
    }
    Ok(lead)
}

/// Every layer's kind from `attention.head_count_kv`: 0 a delta layer, 1 a latent one, as
/// llama.cpp's loader decides it; both kinds are in the file. `kinds` are the family's names for
/// the two.
pub(in crate::arch) fn kinds<K: Copy + PartialEq>(
    split: &Split,
    n_layer: usize,
    (delta, latent): (K, K),
) -> Result<Vec<K>, PlacementError> {
    let key = "attention.head_count_kv";
    let kinds = per_layer_count(split, key, n_layer)?
        .into_iter()
        .enumerate()
        .map(|(l, kv)| match kv {
            0 => Ok(delta),
            1 => Ok(latent),
            other => Err(metadata(
                split,
                key,
                format!("is {other} at layer {l}; 0 marks a delta layer, 1 a latent one"),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let n_delta = kinds.iter().filter(|k| **k == delta).count();
    if n_delta == 0 || n_delta == n_layer {
        return Err(metadata(
            split,
            key,
            format!("has {n_delta} delta layers of {n_layer}; the trunk mixes both kinds"),
        ));
    }
    Ok(kinds)
}
