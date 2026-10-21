//! Every `raon1.*` key the description reads, each layer's mixer from `attention.head_count_kv`,
//! and the tensors the file is held to. A key the loader requires is required here, and a value
//! this reader does not run is refused by the key's name. A layer is an RDA layer when its
//! key-head count is 0 and a latent layer when it is 1; the tensors a layer carries must be
//! exactly its kind's ([`Hparams::read`]). The dialect's authority is llama.cpp's
//! `src/models/raon1.cpp`.

use gguf::Split;
use models::KdaGate;

use super::roles::{KINDS, stems_of};
use crate::arch::hybrid::{layers_agree, no_nextn};
use crate::arch::noaux::{self, Experts};
use crate::arch::{
    dims_agree, finite_positive, kda, meta_usize, metadata, n_vocab, optional_f32, per_layer_count,
    positive, unread_keys_of,
};
use crate::placement::PlacementError;

/// What a layer mixes with, from `attention.head_count_kv`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// 0 key heads: a Raon delta-attention layer.
    Rda,
    /// 1 key head: a gated latent-attention layer without a rope.
    Mla,
}

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
    /// (`attention.key_length`).
    #[must_use]
    pub fn row(&self) -> usize {
        self.kv_lora + self.pe
    }
}

/// The hyper-connection (Raon_mHC) keys.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hc {
    /// `hyper_connection.count`: residual streams.
    pub streams: usize,
    /// `hyper_connection.epsilon`: added to the streams' mean square before the selector's scale.
    pub eps: f32,
    /// `hyper_connection.magnitude`: the post mix's factor.
    pub magnitude: f32,
}

impl Hc {
    /// A sublayer's selector width: `streams` pre mixes, `streams` post mixes and the residual
    /// mix's `(streams - 1)^2` chart coordinates.
    #[must_use]
    pub fn mix_dim(&self) -> usize {
        2 * self.streams + (self.streams - 1) * (self.streams - 1)
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

/// The hyperparameters of one raon1 file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// `block_count`: every layer is a trunk layer, the file carries no next-token block.
    pub n_layer: usize,
    /// `embedding_length`.
    pub n_embd: usize,
    /// `attention.head_count`: the latent layers' query heads and the delta layers' key and
    /// value heads.
    pub n_head: usize,
    /// `ssm.conv_kernel` and `kda.head_dim`: the delta layers' conv taps and head width.
    pub kda: kda::Keys,
    /// `ssm.time_step_rank`: the delta layers' decay's low-rank width.
    pub decay_rank: usize,
    /// How the delta layers' decay is made: `kda.gate_lower_bound` when the file has it (the
    /// sigmoid form bounded below by it), else the softplus form.
    pub gate: KdaGate,
    pub mla: Mla,
    /// `attention.layer_norm_rms_epsilon`: every block norm's.
    pub rms_eps: f32,
    /// `attention.latent_norm_rms_epsilon`: the two latents' norms'.
    pub latent_eps: f32,
    pub hc: Hc,
    pub situ: Situ,
    /// `vocab_size`, else the token list's length.
    pub n_vocab: usize,
    /// `context_length`.
    pub n_ctx_train: usize,
    /// `leading_dense_block_count`: the layers before the first routed one.
    pub dense_lead: usize,
    /// `feed_forward_length`, the dense layers' width.
    pub dense_ff: usize,
    /// The routed layers' router and experts.
    pub experts: Experts,
    /// Every layer's kind, from `attention.head_count_kv`.
    pub kinds: Vec<Kind>,
    /// The keys this read took a default for: the key, the value, the line.
    pub defaults: Vec<String>,
}

/// Every `raon1.` key [`Hparams::read`] reads but the routed block's ([`noaux::KEYS`]), by
/// suffix. The delta rule's `ssm.conv_kernel` and `kda.head_dim` are [`kda::read`]'s.
const READ: &[&str] = &[
    "block_count",
    "nextn_predict_layers",
    "context_length",
    "embedding_length",
    "vocab_size",
    "feed_forward_length",
    "leading_dense_block_count",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.key_length_mla",
    "attention.value_length_mla",
    "attention.q_lora_rank",
    "attention.kv_lora_rank",
    "attention.layer_norm_rms_epsilon",
    "attention.latent_norm_rms_epsilon",
    "rope.dimension_count",
    "ssm.conv_kernel",
    "ssm.state_size",
    "ssm.group_count",
    "ssm.time_step_rank",
    "kda.head_dim",
    "kda.gate_lower_bound",
    "hyper_connection.count",
    "hyper_connection.epsilon",
    "hyper_connection.magnitude",
    "activation.situ_beta",
    "activation.situ_linear_beta",
];

/// The architecture keys and `general.sampling.*` keys of `split` that [`Hparams::read`] does not
/// read: listed, not refused.
#[must_use]
pub fn unread_keys(split: &Split) -> Vec<String> {
    unread_keys_of(split, &[READ, noaux::KEYS].concat())
}

impl Hparams {
    /// Every value from `split`'s headers; a missing required key, a value this reader does not
    /// run, a next-token layer, or a layer whose tensors disagree with its kind or the keys is an
    /// error naming the key or the tensor.
    pub fn read(split: &Split) -> Result<Hparams, PlacementError> {
        let mut defaults = Vec::new();
        let n_layer = positive(split, "block_count")?;
        no_nextn(split)?;
        let n_embd = positive(split, "embedding_length")?;
        let n_head = positive(split, "attention.head_count")?;
        let kinds = kinds(split, n_layer)?;
        let mla = Mla::read(split)?;
        let kda = kda::read(split, n_head, &mut defaults)?;
        let gate = match optional_f32(split, "kda.gate_lower_bound")? {
            Some(lower_bound) if lower_bound.is_finite() && lower_bound < 0.0 => {
                KdaGate::Sigmoid { lower_bound }
            }
            Some(lower_bound) => {
                return Err(metadata(
                    split,
                    "kda.gate_lower_bound",
                    format!("is {lower_bound}; the bounded decay needs a finite bound below 0"),
                ));
            }
            None => {
                defaults.push(
                    "kda.gate_lower_bound = softplus decay (absent, src/models/raon1.cpp)"
                        .to_string(),
                );
                KdaGate::Softplus
            }
        };
        let dense_lead = dense_lead_of(split, n_layer)?;
        let hp = Hparams {
            n_layer,
            n_embd,
            n_head,
            kda,
            decay_rank: positive(split, "ssm.time_step_rank")?,
            gate,
            mla,
            rms_eps: finite_positive(split, "attention.layer_norm_rms_epsilon")?,
            latent_eps: finite_positive(split, "attention.latent_norm_rms_epsilon")?,
            hc: Hc::read(split)?,
            situ: Situ::read(split)?,
            n_vocab: n_vocab(split)?,
            n_ctx_train: positive(split, "context_length")?,
            dense_lead,
            dense_ff: positive(split, "feed_forward_length")?,
            experts: Experts::read(split, "the raon1 router", &mut defaults)?,
            kinds,
            defaults,
        };
        hp.tensors_agree(split)?;
        dims_agree(
            split,
            hp.n_embd,
            hp.n_vocab,
            (0..hp.n_layer).map(|l| hp.layer_dims(l)),
        )?;
        Ok(hp)
    }

    /// Layer `l`'s kind.
    #[must_use]
    pub fn kind(&self, l: usize) -> Kind {
        self.kinds[l]
    }

    /// Whether layer `l` is a dense block, not a routed one.
    #[must_use]
    pub fn dense(&self, l: usize) -> bool {
        l < self.dense_lead
    }

    /// The delta layers' inner width: the heads' channels.
    #[must_use]
    pub fn inner(&self) -> usize {
        self.n_head * self.kda.head_dim
    }

    /// The model-level three and every layer's kind's required tensors are in the file, and
    /// nothing of the other kind's: the first missing or misplaced name, with how many more
    /// there are.
    fn tensors_agree(&self, split: &Split) -> Result<(), PlacementError> {
        let layers: Vec<(Kind, bool)> = (0..self.n_layer)
            .map(|l| (self.kind(l), self.dense(l)))
            .collect();
        layers_agree(split, &layers, stems_of, &KINDS)
    }

    /// Layer `l`'s tensors by stem and the dims llama.cpp creates them with, ggml order.
    #[must_use]
    pub fn layer_dims(&self, l: usize) -> Vec<(&'static str, Vec<u64>)> {
        let (e, h) = (self.n_embd as u64, self.n_head as u64);
        let mut dims: Vec<(&'static str, Vec<u64>)> = match self.kind(l) {
            Kind::Rda => {
                let inner = self.inner() as u64;
                let mut dims = kda::head_dims(
                    self.n_embd,
                    self.n_head,
                    self.kda.head_dim,
                    self.decay_rank,
                    self.kda.conv,
                )
                .to_vec();
                dims.extend([
                    ("ssm_g.weight", vec![e, inner]),
                    ("ssm_rda.weight", vec![e, h]),
                ]);
                dims.extend(kda::tail_dims(self.n_embd, self.n_head, self.kda.head_dim));
                dims
            }
            Kind::Mla => {
                let m = &self.mla;
                let (q, kv, k, v) = (
                    m.q_lora as u64,
                    m.kv_lora as u64,
                    m.head_k as u64,
                    m.head_v as u64,
                );
                vec![
                    ("attn_norm.weight", vec![e]),
                    ("attn_q_a.weight", vec![e, q]),
                    ("attn_q_a_norm.weight", vec![q]),
                    ("attn_q_b.weight", vec![q, h * k]),
                    ("attn_kv_a_mqa.weight", vec![e, m.row() as u64]),
                    ("attn_kv_a_norm.weight", vec![kv]),
                    ("attn_k_b.weight", vec![m.nope() as u64, kv, h]),
                    ("attn_v_b.weight", vec![kv, v, h]),
                    ("attn_gate.weight", vec![e, h * v]),
                    ("attn_output.weight", vec![h * v, e]),
                ]
            }
        };
        dims.push(("ffn_norm.weight", vec![e]));
        if self.dense(l) {
            let ff = self.dense_ff as u64;
            dims.extend([
                ("ffn_gate.weight", vec![e, ff]),
                ("ffn_up.weight", vec![e, ff]),
                ("ffn_down.weight", vec![ff, e]),
            ]);
        } else {
            dims.extend(self.experts.dims(e));
        }
        let mix = self.hc.mix_dim() as u64;
        let streams_e = self.hc.streams as u64 * e;
        dims.extend([
            ("hc_attn_fn.weight", vec![streams_e, mix]),
            ("hc_attn_base.weight", vec![mix]),
            ("hc_attn_scale.weight", vec![HC_SCALES]),
            ("hc_ffn_fn.weight", vec![streams_e, mix]),
            ("hc_ffn_base.weight", vec![mix]),
            ("hc_ffn_scale.weight", vec![HC_SCALES]),
        ]);
        dims
    }
}

/// A sublayer's mix scalars: `alpha_pre`, `alpha_post`, `alpha_z` and `eta`.
pub const HC_SCALES: u64 = 4;

impl Mla {
    fn read(split: &Split) -> Result<Mla, PlacementError> {
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
        // The cache holds the latent and the unrotated values as the key, and the latent as the
        // value: the generic key and value lengths say so, and the body reads the rows by them.
        for (key, want, what) in [
            (
                "attention.key_length",
                mla.row(),
                "kv_lora_rank + rope.dimension_count",
            ),
            ("attention.value_length", mla.kv_lora, "kv_lora_rank"),
        ] {
            let got = positive(split, key)?;
            if got != want {
                return Err(metadata(
                    split,
                    key,
                    format!(
                        "is {got}, not {what} ({want}): the cached row is the latent and its unrotated values"
                    ),
                ));
            }
        }
        Ok(mla)
    }
}

impl Hc {
    fn read(split: &Split) -> Result<Hc, PlacementError> {
        let streams = positive(split, "hyper_connection.count")?;
        if streams < 2 {
            return Err(metadata(
                split,
                "hyper_connection.count",
                format!("is {streams}; the chart mix needs at least two streams"),
            ));
        }
        Ok(Hc {
            streams,
            eps: finite_positive(split, "hyper_connection.epsilon")?,
            magnitude: finite_positive(split, "hyper_connection.magnitude")?,
        })
    }
}

impl Situ {
    fn read(split: &Split) -> Result<Situ, PlacementError> {
        Ok(Situ {
            gate_beta: finite_positive(split, "activation.situ_beta")?,
            value_beta: finite_positive(split, "activation.situ_linear_beta")?,
        })
    }
}

/// `leading_dense_block_count`, required, below the layer count: a trunk with no routed layer
/// is another reader's decision.
fn dense_lead_of(split: &Split, n_layer: usize) -> Result<usize, PlacementError> {
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

/// Every layer's kind from `attention.head_count_kv`: 0 an RDA layer, 1 a latent one, as
/// llama.cpp's loader decides it; both kinds are in the file.
fn kinds(split: &Split, n_layer: usize) -> Result<Vec<Kind>, PlacementError> {
    let key = "attention.head_count_kv";
    let kinds = per_layer_count(split, key, n_layer)?
        .into_iter()
        .enumerate()
        .map(|(l, kv)| match kv {
            0 => Ok(Kind::Rda),
            1 => Ok(Kind::Mla),
            other => Err(metadata(
                split,
                key,
                format!("is {other} at layer {l}; 0 marks a delta layer, 1 a latent one"),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let rda = kinds.iter().filter(|k| **k == Kind::Rda).count();
    if rda == 0 || rda == n_layer {
        return Err(metadata(
            split,
            key,
            format!("has {rda} delta layers of {n_layer}; the trunk mixes both kinds"),
        ));
    }
    Ok(kinds)
}

#[cfg(test)]
pub(super) mod tests {
    use super::{Hparams, Kind, unread_keys};
    use crate::arch::synthetic::{V, header_shaped};
    use gguf::Split;
    use models::KdaGate;

    /// The widths of [`keys`]: 6 layers, delta layers at 0, 1, 3 and 4 and latent ones at 2 and 5,
    /// the first dense; 64 wide, 4 heads, delta heads of 16 with a decay of rank 8; latent
    /// ranks 24 and 32, a head of 16 + 8 query values and 16 value values; dense 48; 8 experts of
    /// 24, 2 used, 1 shared; 3 streams.
    pub(in crate::arch::raon1) const E: u64 = 64;
    pub(in crate::arch::raon1) const HEADS: u64 = 4;
    pub(in crate::arch::raon1) const D: u64 = 16;
    pub(in crate::arch::raon1) const CONV: u64 = 4;
    pub(in crate::arch::raon1) const RANK: u64 = 8;
    pub(in crate::arch::raon1) const Q_LORA: u64 = 24;
    pub(in crate::arch::raon1) const KV_LORA: u64 = 32;
    pub(in crate::arch::raon1) const NOPE: u64 = 16;
    pub(in crate::arch::raon1) const PE: u64 = 8;
    pub(in crate::arch::raon1) const HEAD_V: u64 = 16;
    pub(in crate::arch::raon1) const LAYERS: usize = 6;
    pub(in crate::arch::raon1) const MLA_LAYERS: [usize; 2] = [2, 5];
    pub(in crate::arch::raon1) const FF: u64 = 48;
    pub(in crate::arch::raon1) const N_EXP: u64 = 8;
    pub(in crate::arch::raon1) const EFF: u64 = 24;
    pub(in crate::arch::raon1) const STREAMS: u64 = 3;
    /// The two-token vocabulary of the synthetic header.
    pub(in crate::arch::raon1) const VOCAB: u64 = 2;

    /// The keys of the released file, at this table's widths.
    pub(in crate::arch::raon1) fn keys() -> Vec<(&'static str, V)> {
        let kv: Vec<i32> = (0..LAYERS)
            .map(|l| i32::from(MLA_LAYERS.contains(&l)))
            .collect();
        vec![
            ("block_count", V::U32(LAYERS as u32)),
            ("context_length", V::U32(4096)),
            ("embedding_length", V::U32(E as u32)),
            ("feed_forward_length", V::U32(FF as u32)),
            ("leading_dense_block_count", V::U32(1)),
            ("attention.head_count", V::U32(HEADS as u32)),
            ("attention.head_count_kv", V::I32s(kv)),
            ("attention.key_length", V::U32((KV_LORA + PE) as u32)),
            ("attention.value_length", V::U32(KV_LORA as u32)),
            ("attention.key_length_mla", V::U32((NOPE + PE) as u32)),
            ("attention.value_length_mla", V::U32(HEAD_V as u32)),
            ("attention.q_lora_rank", V::U32(Q_LORA as u32)),
            ("attention.kv_lora_rank", V::U32(KV_LORA as u32)),
            ("attention.layer_norm_rms_epsilon", V::F32(1e-5)),
            ("attention.latent_norm_rms_epsilon", V::F32(1e-6)),
            ("rope.dimension_count", V::U32(PE as u32)),
            ("ssm.conv_kernel", V::U32(CONV as u32)),
            ("ssm.time_step_rank", V::U32(RANK as u32)),
            ("kda.head_dim", V::U32(D as u32)),
            ("kda.gate_lower_bound", V::F32(-5.0)),
            ("expert_count", V::U32(N_EXP as u32)),
            ("expert_used_count", V::U32(2)),
            ("expert_feed_forward_length", V::U32(EFF as u32)),
            ("expert_shared_count", V::U32(1)),
            ("expert_gating_func", V::U32(2)),
            ("expert_weights_scale", V::F32(1.0)),
            ("expert_weights_norm", V::Bool(true)),
            ("hyper_connection.count", V::U32(STREAMS as u32)),
            ("hyper_connection.epsilon", V::F32(1e-5)),
            ("hyper_connection.magnitude", V::F32(2.0)),
            ("activation.situ_beta", V::F32(4.0)),
            ("activation.situ_linear_beta", V::F32(25.0)),
        ]
    }

    /// The released file's tensor table (731 tensors at 25 layers), at [`keys`]' widths: the three
    /// model-level tensors, then each layer's mixer, feed-forward block and mixes. Written from the
    /// table, apart from the reader.
    pub(in crate::arch::raon1) fn tensors() -> Vec<(String, Vec<u64>)> {
        let mut t: Vec<(String, Vec<u64>)> = vec![
            ("token_embd.weight".into(), vec![E, VOCAB]),
            ("output.weight".into(), vec![E, VOCAB]),
            ("output_norm.weight".into(), vec![E]),
        ];
        let inner = HEADS * D;
        let mix = 2 * STREAMS + (STREAMS - 1) * (STREAMS - 1);
        for l in 0..LAYERS {
            let mut layer: Vec<(&str, Vec<u64>)> = vec![("attn_norm.weight", vec![E])];
            if MLA_LAYERS.contains(&l) {
                layer.extend([
                    ("attn_q_a.weight", vec![E, Q_LORA]),
                    ("attn_q_a_norm.weight", vec![Q_LORA]),
                    ("attn_q_b.weight", vec![Q_LORA, HEADS * (NOPE + PE)]),
                    ("attn_kv_a_mqa.weight", vec![E, KV_LORA + PE]),
                    ("attn_kv_a_norm.weight", vec![KV_LORA]),
                    ("attn_k_b.weight", vec![NOPE, KV_LORA, HEADS]),
                    ("attn_v_b.weight", vec![KV_LORA, HEAD_V, HEADS]),
                    ("attn_gate.weight", vec![E, HEADS * HEAD_V]),
                    ("attn_output.weight", vec![HEADS * HEAD_V, E]),
                ]);
            } else {
                layer.extend([
                    ("attn_q.weight", vec![E, inner]),
                    ("attn_k.weight", vec![E, inner]),
                    ("attn_v.weight", vec![E, inner]),
                    ("ssm_conv1d_q.weight", vec![CONV, 1, inner]),
                    ("ssm_conv1d_k.weight", vec![CONV, 1, inner]),
                    ("ssm_conv1d_v.weight", vec![CONV, 1, inner]),
                    ("ssm_f_a.weight", vec![E, RANK]),
                    ("ssm_f_b.weight", vec![RANK, inner]),
                    ("ssm_beta.weight", vec![E, HEADS]),
                    ("ssm_rda.weight", vec![E, HEADS]),
                    ("ssm_a", vec![HEADS]),
                    ("ssm_dt.bias", vec![inner]),
                    ("ssm_g.weight", vec![E, inner]),
                    ("ssm_norm.weight", vec![D]),
                    ("attn_output.weight", vec![inner, E]),
                ]);
            }
            layer.push(("ffn_norm.weight", vec![E]));
            if l == 0 {
                layer.extend([
                    ("ffn_gate.weight", vec![E, FF]),
                    ("ffn_up.weight", vec![E, FF]),
                    ("ffn_down.weight", vec![FF, E]),
                ]);
            } else {
                layer.extend([
                    ("ffn_gate_inp.weight", vec![E, N_EXP]),
                    ("exp_probs_b.bias", vec![N_EXP]),
                    ("ffn_gate_exps.weight", vec![E, EFF, N_EXP]),
                    ("ffn_up_exps.weight", vec![E, EFF, N_EXP]),
                    ("ffn_down_exps.weight", vec![EFF, E, N_EXP]),
                    ("ffn_gate_shexp.weight", vec![E, EFF]),
                    ("ffn_up_shexp.weight", vec![E, EFF]),
                    ("ffn_down_shexp.weight", vec![EFF, E]),
                ]);
            }
            for sub in ["attn", "ffn"] {
                layer.push((leak(format!("hc_{sub}_fn.weight")), vec![STREAMS * E, mix]));
                layer.push((leak(format!("hc_{sub}_base.weight")), vec![mix]));
                layer.push((leak(format!("hc_{sub}_scale.weight")), vec![4]));
            }
            t.extend(layer.into_iter().map(|(s, d)| (format!("blk.{l}.{s}"), d)));
        }
        t
    }

    /// A name that lives as long as the process: a test table's computed stem.
    fn leak(s: String) -> &'static str {
        Box::leak(s.into_boxed_str())
    }

    /// [`keys`] with `key` replaced by `v`, or dropped when `v` is `None`.
    pub(in crate::arch::raon1) fn with(key: &'static str, v: Option<V>) -> Vec<(&'static str, V)> {
        let mut kv: Vec<_> = keys().into_iter().filter(|(k, _)| *k != key).collect();
        if let Some(v) = v {
            kv.push((key, v));
        }
        kv
    }

    /// The reader's verdict on a header of `kv` and `tensors`, as the error text.
    fn read(
        tag: &str,
        kv: &[(&str, V)],
        tensors: &[(String, Vec<u64>)],
    ) -> Result<Hparams, String> {
        let path = header_shaped(&format!("raon1-hparams-{tag}"), "raon1", kv, &[], tensors);
        let split = Split::open(&path).map_err(|e| e.to_string());
        let got = split.and_then(|s| Hparams::read(&s).map_err(|e| e.to_string()));
        let _ = std::fs::remove_file(&path);
        got
    }

    /// One refusal case: a tag, the header's keys, the tensor shapes and the name the refusal must hold.
    type Case = (
        String,
        Vec<(&'static str, V)>,
        Vec<(String, Vec<u64>)>,
        &'static str,
    );

    /// What is wrong with the verdict on `tag`'s header: it read, or the error lacks `want`.
    fn wrong_verdict(
        tag: &str,
        kv: &[(&str, V)],
        tensors: &[(String, Vec<u64>)],
        want: &str,
    ) -> Option<String> {
        match read(tag, kv, tensors) {
            Ok(_) => Some(format!("{tag}: read; expected {want:?}")),
            Err(e) if e.contains(want) => None,
            Err(e) => Some(format!("{tag}: want {want:?} in: {e}")),
        }
    }

    /// The tensors with `name` dropped.
    fn without(name: &str) -> Vec<(String, Vec<u64>)> {
        tensors().into_iter().filter(|(n, _)| n != name).collect()
    }

    /// The tensors with `name` given `dims`, or added when the table has none.
    fn shaped(name: &str, dims: Vec<u64>) -> Vec<(String, Vec<u64>)> {
        let mut t = without(name);
        t.push((name.to_string(), dims));
        t
    }

    /// The header the table gives reads, to the values its keys hold, with the head counts
    /// splitting the layers and no default taken.
    #[test]
    fn a_small_header_reads() {
        let hp = read("small", &keys(), &tensors()).expect("the header reads");
        assert_eq!((hp.n_layer, hp.n_embd, hp.n_head), (LAYERS, 64, 4));
        assert_eq!((hp.kda.conv, hp.kda.head_dim, hp.decay_rank), (4, 16, 8));
        assert_eq!(hp.gate, KdaGate::Sigmoid { lower_bound: -5.0 });
        assert_eq!(
            (
                hp.mla.q_lora,
                hp.mla.kv_lora,
                hp.mla.head_k,
                hp.mla.head_v,
                hp.mla.pe
            ),
            (24, 32, 24, 16, 8)
        );
        assert_eq!((hp.mla.nope(), hp.mla.row()), (16, 40));
        assert_eq!((hp.rms_eps, hp.latent_eps), (1e-5, 1e-6));
        assert_eq!((hp.hc.streams, hp.hc.eps, hp.hc.magnitude), (3, 1e-5, 2.0));
        assert_eq!(hp.hc.mix_dim(), 10);
        assert_eq!((hp.situ.gate_beta, hp.situ.value_beta), (4.0, 25.0));
        assert_eq!(
            (hp.n_vocab, hp.n_ctx_train, hp.dense_lead, hp.dense_ff),
            (2, 4096, 1, 48)
        );
        assert_eq!(
            (
                hp.experts.n_expert,
                hp.experts.n_used,
                hp.experts.ff,
                hp.experts.shared_ff()
            ),
            (8, 2, 24, 24)
        );
        assert_eq!(
            (hp.experts.weights_scale, hp.experts.weights_norm),
            (1.0, true)
        );
        assert_eq!(hp.inner(), 64);
        let kinds: Vec<Kind> = (0..LAYERS).map(|l| hp.kind(l)).collect();
        assert_eq!(
            kinds,
            [
                Kind::Rda,
                Kind::Rda,
                Kind::Mla,
                Kind::Rda,
                Kind::Rda,
                Kind::Mla
            ]
        );
        assert!(hp.dense(0) && !hp.dense(1));
        assert!(hp.defaults.is_empty(), "{:?}", hp.defaults);
    }

    /// An absent `kda.gate_lower_bound` is the softplus decay, recorded as a default; a bound
    /// that is not below 0 is refused by its key.
    #[test]
    fn the_decay_gate_follows_its_bound() {
        let hp = read("no-bound", &with("kda.gate_lower_bound", None), &tensors()).expect("reads");
        assert_eq!(hp.gate, KdaGate::Softplus);
        assert!(
            hp.defaults
                .iter()
                .any(|d| d.contains("kda.gate_lower_bound")),
            "{:?}",
            hp.defaults
        );
        for (tag, bound) in [("zero", 0.0_f32), ("up", 5.0), ("nan", f32::NAN)] {
            let kv = with("kda.gate_lower_bound", Some(V::F32(bound)));
            assert_eq!(
                wrong_verdict(tag, &kv, &tensors(), "kda.gate_lower_bound"),
                None
            );
        }
    }

    /// Every refusal names its key or tensor. Mutants: a check dropped, or the wrong name given.
    #[test]
    fn the_reader_refuses_by_name() {
        let cases: Vec<Case> = vec![
            (
                "no-rank".into(),
                with("ssm.time_step_rank", None),
                tensors(),
                "ssm.time_step_rank",
            ),
            (
                "no-hc".into(),
                with("hyper_connection.count", None),
                tensors(),
                "hyper_connection.count",
            ),
            (
                "one-stream".into(),
                with("hyper_connection.count", Some(V::U32(1))),
                tensors(),
                "at least two streams",
            ),
            (
                "no-magnitude".into(),
                with("hyper_connection.magnitude", None),
                tensors(),
                "hyper_connection.magnitude",
            ),
            (
                "no-situ".into(),
                with("activation.situ_beta", None),
                tensors(),
                "activation.situ_beta",
            ),
            (
                "zero-situ".into(),
                with("activation.situ_linear_beta", Some(V::F32(0.0))),
                tensors(),
                "activation.situ_linear_beta",
            ),
            (
                "no-latent-eps".into(),
                with("attention.latent_norm_rms_epsilon", None),
                tensors(),
                "attention.latent_norm_rms_epsilon",
            ),
            (
                "no-pe".into(),
                with("rope.dimension_count", None),
                tensors(),
                "rope.dimension_count",
            ),
            (
                "pe-fills-head".into(),
                with("rope.dimension_count", Some(V::U32(24))),
                tensors(),
                "attention.key_length_mla",
            ),
            (
                "row-width".into(),
                with("attention.key_length", Some(V::U32(576))),
                tensors(),
                "attention.key_length",
            ),
            (
                "value-width".into(),
                with("attention.value_length", Some(V::U32(16))),
                tensors(),
                "attention.value_length",
            ),
            (
                "softmax".into(),
                with("expert_gating_func", Some(V::U32(1))),
                tensors(),
                "expert_gating_func",
            ),
            (
                "lead-all".into(),
                with("leading_dense_block_count", Some(V::U32(6))),
                tensors(),
                "leading_dense_block_count",
            ),
            (
                "no-lead".into(),
                with("leading_dense_block_count", None),
                tensors(),
                "leading_dense_block_count",
            ),
            (
                "nextn".into(),
                with("nextn_predict_layers", Some(V::U32(1))),
                tensors(),
                "nextn_predict_layers",
            ),
            (
                "kv-two".into(),
                with(
                    "attention.head_count_kv",
                    Some(V::I32s(vec![0, 0, 2, 0, 0, 1])),
                ),
                tensors(),
                "2 at layer 2",
            ),
            (
                "all-delta".into(),
                with("attention.head_count_kv", Some(V::I32s(vec![0; 6]))),
                tensors(),
                "6 delta layers of 6",
            ),
            (
                "all-latent".into(),
                with("attention.head_count_kv", Some(V::I32s(vec![1; 6]))),
                tensors(),
                "0 delta layers of 6",
            ),
            (
                "kv-short".into(),
                with("attention.head_count_kv", Some(V::I32s(vec![0, 1]))),
                tensors(),
                "attention.head_count_kv",
            ),
            (
                "no-gate".into(),
                keys(),
                without("blk.2.attn_gate.weight"),
                "blk.2.attn_gate.weight",
            ),
            (
                "no-write-gate".into(),
                keys(),
                without("blk.1.ssm_rda.weight"),
                "blk.1.ssm_rda.weight",
            ),
            (
                "no-hc-fn".into(),
                keys(),
                without("blk.4.hc_ffn_fn.weight"),
                "blk.4.hc_ffn_fn.weight",
            ),
            (
                "no-embd".into(),
                keys(),
                without("token_embd.weight"),
                "token_embd.weight",
            ),
            (
                "latent-on-delta".into(),
                keys(),
                shaped("blk.1.attn_gate.weight", vec![E, HEADS * HEAD_V]),
                "blk.1.attn_gate.weight",
            ),
            (
                "delta-on-latent".into(),
                keys(),
                shaped("blk.2.ssm_rda.weight", vec![E, HEADS]),
                "blk.2.ssm_rda.weight",
            ),
            (
                "dense-with-router".into(),
                keys(),
                shaped("blk.0.ffn_gate_inp.weight", vec![E, N_EXP]),
                "blk.0.ffn_gate_inp.weight",
            ),
            (
                "routed-with-dense".into(),
                keys(),
                shaped("blk.3.ffn_gate.weight", vec![E, FF]),
                "blk.3.ffn_gate.weight",
            ),
            (
                "decay-rank".into(),
                keys(),
                shaped("blk.1.ssm_f_b.weight", vec![RANK + 1, HEADS * D]),
                "blk.1.ssm_f_b.weight",
            ),
            (
                "latent-row".into(),
                keys(),
                shaped("blk.5.attn_kv_a_mqa.weight", vec![E, KV_LORA]),
                "blk.5.attn_kv_a_mqa.weight",
            ),
            (
                "hc-mix".into(),
                keys(),
                shaped("blk.3.hc_attn_fn.weight", vec![STREAMS * E, 9]),
                "blk.3.hc_attn_fn.weight",
            ),
            (
                "hc-scale".into(),
                keys(),
                shaped("blk.3.hc_ffn_scale.weight", vec![3]),
                "blk.3.hc_ffn_scale.weight",
            ),
            (
                "k-b".into(),
                keys(),
                shaped("blk.5.attn_k_b.weight", vec![NOPE, KV_LORA, 1]),
                "blk.5.attn_k_b.weight",
            ),
        ];
        let wrong: Vec<String> = cases
            .iter()
            .filter_map(|(tag, kv, t, want)| wrong_verdict(tag, kv, t, want))
            .collect();
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    /// The reader lists, and does not refuse, a key it does not read.
    #[test]
    fn an_unread_key_is_listed() {
        let mut kv = keys();
        kv.push(("rope.freq_base", V::F32(10_000.0)));
        let path = header_shaped("raon1-hparams-unread", "raon1", &kv, &[], &tensors());
        let split = Split::open(&path).expect("opens");
        let unread = unread_keys(&split);
        let _ = std::fs::remove_file(&path);
        assert_eq!(unread, vec!["raon1.rope.freq_base".to_string()]);
    }
}
