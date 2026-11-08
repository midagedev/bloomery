//! Every `kimi-k3.*` and `kimi-k31.*` key the description reads, each layer's mixer from
//! `attention.head_count_kv`, and the tensors the file is held to: the names every layer's kind
//! carries, their dims and the quant types the engine reads them in. A key the loader requires is
//! required here, and a value this reader does not run is refused by the key's name. A layer is a
//! KDA layer when its key-head count is 0 and a latent layer when it is 1; the tensors a layer
//! carries must be exactly its kind's ([`Hparams::read`]). A `kimi-k31` file is a `kimi-k3` one
//! whose every latent layer carries the lightning indexer of DeepSeek's sparse attention (the
//! `attention.indexer.*` keys and five tensors a layer, [`Indexer`]). The dialect's authority is
//! llama.cpp's `src/models/kimi-k3.cpp`; K3.1 has none, so its keys are this engine's, spelled as
//! Mistral Large 4's indexer keys are where the meaning is the same.

use gguf::Split;
use gguf::arch::kimi_k3 as released;
use models::KdaGate;

use super::names::OUTPUT_RES_SCORE;
use super::roles::{KINDS, stems_of};
use crate::arch::classify::block_stem;
use crate::arch::delta_mla::{self, Mla, Situ};
use crate::arch::hybrid::{layers_agree, no_nextn};
use crate::arch::noaux::{self, Experts};
use crate::arch::{
    architecture_is, dims_agree, finite_positive, kda, meta_usize, metadata, n_vocab,
    per_layer_bool, positive, unread_keys_of,
};
use crate::placement::PlacementError;

/// What a layer mixes with, from `attention.head_count_kv`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// 0 key heads: a Kimi delta-attention layer.
    Kda,
    /// 1 key head: a gated latent-attention layer without a rope.
    Mla,
}

/// The index heads the score kernel runs: the lifted indexer's `HEADS` (Mistral Large 4's reader
/// holds the same constant), so a file that names another is refused rather than planned for a
/// launch that does not exist.
pub const INDEX_HEADS: usize = 32;

/// The index head's width: the key kernel's Hadamard is the 128-wide one, and the score kernel
/// reads a key of this many values.
pub const INDEX_DIM: usize = 128;

/// The lightning indexer of a K3.1 file: on each latent layer, a query from the query latent
/// (`indexer.attn_q_b`, the RMS-normed `attn_q_a` output to `heads` heads of `d` values), a key
/// from the layer input (`indexer.attn_k`, then a LayerNorm with gain and bias at `norm_eps`, no
/// rope, stored f16 for every position), and per-head weights from the layer input
/// (`indexer.proj`); a position's score is the heads' weighted sum of the rectified dots, and the
/// layer attends the `top_k` best positions. At a context of `top_k` positions or fewer the
/// selection is every position.
#[derive(Clone, Debug, PartialEq)]
pub struct Indexer {
    /// `attention.indexer.head_count`: [`INDEX_HEADS`].
    pub heads: usize,
    /// `attention.indexer.key_length`: [`INDEX_DIM`].
    pub d: usize,
    /// `attention.indexer.top_k`: the positions a layer attends past that many.
    pub top_k: usize,
    /// `attention.layer_norm_epsilon`: the key LayerNorm's.
    pub norm_eps: f32,
    /// `attention.indexer.types`: whether each layer runs the indexer, the latent layers and no
    /// other.
    pub layers: Vec<bool>,
}

/// The hyperparameters of one Kimi K3 or K3.1 file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// The file's architecture word: [`released::ARCH`] or [`released::ARCH_K31`].
    pub arch: &'static str,
    /// `block_count`: every layer is a trunk layer, the file carries no next-token block.
    pub n_layer: usize,
    /// `embedding_length`.
    pub n_embd: usize,
    /// `attention.head_count`: the latent layers' query heads and the delta layers' key and
    /// value heads.
    pub n_head: usize,
    /// `ssm.conv_kernel` and `kda.head_dim`: the delta layers' conv taps and head width. The
    /// decay's low-rank width is the head width.
    pub kda: kda::Keys,
    /// How the delta layers' decay is made: `kda.gate_lower_bound` when the file has it (the
    /// sigmoid form bounded below by it), else the softplus form.
    pub gate: KdaGate,
    pub mla: Mla,
    /// `attention.layer_norm_rms_epsilon`: every block norm's, the routed sum's among them.
    pub rms_eps: f32,
    pub situ: Situ,
    /// `expert_latent_length`: the width the routed experts run in.
    pub latent: usize,
    /// `attn_res.block_size`: the layers between two banks of the cross-layer residual.
    pub attn_res_block: usize,
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
    /// The K3.1 file's indexer; `None` for a K3 file.
    pub indexer: Option<Indexer>,
    /// The keys this read took a default for: the key, the value, the line.
    pub defaults: Vec<String>,
}

/// Every `kimi-k3.` or `kimi-k31.` key [`Hparams::read`] reads but the routed block's ([`noaux::KEYS`]), by
/// suffix. The delta rule's `ssm.conv_kernel`, `kda.head_dim`, `ssm.state_size` and
/// `ssm.group_count` are [`kda::read`]'s. `attention.value_length` is not read: the file's value
/// does not describe the cache (the value is the key row's latent), and the reader takes the
/// widths from the latent keys.
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
    "attention.key_length_mla",
    "attention.value_length_mla",
    "attention.q_lora_rank",
    "attention.kv_lora_rank",
    "attention.layer_norm_rms_epsilon",
    "rope.dimension_count",
    "ssm.conv_kernel",
    "ssm.state_size",
    "ssm.group_count",
    "kda.head_dim",
    "kda.gate_lower_bound",
    "expert_latent_length",
    "attn_res.block_size",
    "activation.situ_beta",
    "activation.situ_linear_beta",
];

/// The keys of a K3.1 file's indexer, by suffix. `attention.indexer.head_count`, `.key_length`,
/// `.top_k`, `.types` and `attention.layer_norm_epsilon` are Mistral Large 4's spellings for the
/// same meanings; `attention.indexer.rope_dimension_count` is this family's own, since there the
/// indexer's rope is the layer's `rope.dimension_count` and K3.1's layers rotate nothing.
const INDEXER_KEYS: &[&str] = &[
    "attention.indexer.head_count",
    "attention.indexer.key_length",
    "attention.indexer.top_k",
    "attention.indexer.types",
    "attention.indexer.rope_dimension_count",
    "attention.layer_norm_epsilon",
];

/// The architecture keys and `general.sampling.*` keys of `split` that [`Hparams::read`] does not
/// read: listed, not refused.
#[must_use]
pub fn unread_keys(split: &Split) -> Vec<String> {
    let indexer: &[&str] = if split.architecture() == Some(released::ARCH_K31) {
        INDEXER_KEYS
    } else {
        &[]
    };
    unread_keys_of(split, &[READ, noaux::KEYS, indexer].concat())
}

impl Hparams {
    /// Every value from `split`'s headers; a missing required key, a value this reader does not
    /// run, a next-token layer, or a layer whose tensors disagree with its kind, the keys or the
    /// released set's types is an error naming the key or the tensor.
    pub fn read(split: &Split) -> Result<Hparams, PlacementError> {
        let arch = match split.architecture() {
            Some(released::ARCH) => released::ARCH,
            Some(released::ARCH_K31) => released::ARCH_K31,
            other => {
                return Err(architecture_is(&format!(
                    "{:?}; this reader reads {:?} and {:?} files",
                    other.unwrap_or("<missing>"),
                    released::ARCH,
                    released::ARCH_K31
                )));
            }
        };
        let mut defaults = Vec::new();
        let n_layer = positive(split, "block_count")?;
        no_nextn(split)?;
        let n_embd = positive(split, "embedding_length")?;
        let n_head = positive(split, "attention.head_count")?;
        let kinds = delta_mla::kinds(split, n_layer, (Kind::Kda, Kind::Mla))?;
        let mla = Mla::read(split)?;
        let indexer = (arch == released::ARCH_K31)
            .then(|| Indexer::read(split, &kinds))
            .transpose()?;
        positive(split, "kda.head_dim")?;
        let kda = kda::read(split, n_head, &mut defaults)?;
        let gate = kda::gate(split, "src/models/kimi-k3.cpp", &mut defaults)?;
        let dense_lead = delta_mla::dense_lead_of(split, n_layer)?;
        let hp = Hparams {
            arch,
            n_layer,
            n_embd,
            n_head,
            kda,
            gate,
            mla,
            rms_eps: finite_positive(split, "attention.layer_norm_rms_epsilon")?,
            situ: Situ::read(split)?,
            latent: positive(split, "expert_latent_length")?,
            attn_res_block: positive(split, "attn_res.block_size")?,
            n_vocab: n_vocab(split)?,
            n_ctx_train: positive(split, "context_length")?,
            dense_lead,
            dense_ff: positive(split, "feed_forward_length")?,
            experts: Experts::read(split, "the kimi-k3 router", &mut defaults)?,
            kinds,
            indexer,
            defaults,
        };
        hp.tensors_agree(split)?;
        dims_agree(
            split,
            hp.n_embd,
            hp.n_vocab,
            (0..hp.n_layer).map(|l| hp.layer_dims(l)),
        )?;
        hp.final_mix_agrees(split)?;
        types_agree(split, hp.n_layer)?;
        Ok(hp)
    }

    /// Layer `l`'s kind.
    #[must_use]
    pub fn kind(&self, l: usize) -> Kind {
        self.kinds[l]
    }

    /// Whether layer `l` runs the lightning indexer: a latent layer of a K3.1 file.
    #[must_use]
    pub fn dsa(&self, l: usize) -> bool {
        self.indexer.as_ref().is_some_and(|ix| ix.layers[l])
    }

    /// A context of `ctx` positions the engine runs: every context the plan holds, of a K3 file and
    /// of a K3.1 file alike. Past a K3.1 file's `attention.indexer.top_k` its latent layers attend
    /// the rows their indexer picks, the file's attention at any context; at or under it the
    /// selection is every position and the dense latent attention is its result.
    pub fn check_ctx(&self, _ctx: u64) -> Result<(), PlacementError> {
        Ok(())
    }

    /// Whether layer `l` is a dense block, not a routed one.
    #[must_use]
    pub fn dense(&self, l: usize) -> bool {
        l < self.dense_lead
    }

    /// Whether layer `l` banks its input for the cross-layer residual: every
    /// `attn_res.block_size`th layer, the first included.
    #[must_use]
    pub fn banks(&self, l: usize) -> bool {
        l.is_multiple_of(self.attn_res_block)
    }

    /// The delta layers' inner width: the heads' channels.
    #[must_use]
    pub fn inner(&self) -> usize {
        self.n_head * self.kda.head_dim
    }

    /// One delta layer's recurrent state: a `head_dim` by `head_dim` matrix for each head.
    #[must_use]
    pub fn kda_state(&self) -> [usize; 3] {
        [self.n_head, self.kda.head_dim, self.kda.head_dim]
    }

    /// The model-level three and every layer's kind's required tensors are in the file, and
    /// nothing of the other kind's: the first missing or misplaced name, with how many more
    /// there are. The final mix's scores are required too.
    fn tensors_agree(&self, split: &Split) -> Result<(), PlacementError> {
        let layers: Vec<(Kind, bool, bool)> = (0..self.n_layer)
            .map(|l| (self.kind(l), self.dense(l), self.dsa(l)))
            .collect();
        layers_agree(split, &layers, stems_of, &KINDS)?;
        if split.find(OUTPUT_RES_SCORE).is_none() {
            return Err(PlacementError::Tensor {
                name: OUTPUT_RES_SCORE.to_string(),
                detail: "is not in the file (the final mix of the banks needs its scores)"
                    .to_string(),
            });
        }
        Ok(())
    }

    /// The final mix's score vector holds one value for each hidden value.
    fn final_mix_agrees(&self, split: &Split) -> Result<(), PlacementError> {
        let want = vec![self.n_embd as u64];
        match split.find(OUTPUT_RES_SCORE) {
            Some((_, t)) if t.dims != want => Err(PlacementError::Tensor {
                name: OUTPUT_RES_SCORE.to_string(),
                detail: format!("has dims {:?}; the hidden width gives {want:?}", t.dims),
            }),
            _ => Ok(()),
        }
    }

    /// Layer `l`'s tensors by stem and the dims llama.cpp creates them with, ggml order. The
    /// routed stacks run in the latent, the router and the shared expert read the hidden row.
    #[must_use]
    pub fn layer_dims(&self, l: usize) -> Vec<(&'static str, Vec<u64>)> {
        let (e, h) = (self.n_embd as u64, self.n_head as u64);
        let mut dims: Vec<(&'static str, Vec<u64>)> = match self.kind(l) {
            Kind::Kda => {
                let inner = self.inner() as u64;
                let mut dims = kda::head_dims(
                    self.n_embd,
                    self.n_head,
                    self.kda.head_dim,
                    self.kda.head_dim,
                    self.kda.conv,
                )
                .to_vec();
                dims.push(("ssm_g.weight", vec![e, inner]));
                dims.extend(kda::tail_dims(self.n_embd, self.n_head, self.kda.head_dim));
                dims
            }
            Kind::Mla => self.mla.dims(e, h).to_vec(),
        };
        if let Some(ix) = self.indexer.as_ref().filter(|ix| ix.layers[l]) {
            let (heads, d, q_lora) = (ix.heads as u64, ix.d as u64, self.mla.q_lora as u64);
            dims.extend([
                ("indexer.attn_q_b.weight", vec![q_lora, heads * d]),
                ("indexer.attn_k.weight", vec![e, d]),
                ("indexer.proj.weight", vec![e, heads]),
                ("indexer.k_norm.weight", vec![d]),
                ("indexer.k_norm.bias", vec![d]),
            ]);
        }
        dims.extend([
            ("attn_res_score.weight", vec![e]),
            ("ffn_res_score.weight", vec![e]),
            ("ffn_norm.weight", vec![e]),
        ]);
        if self.dense(l) {
            let ff = self.dense_ff as u64;
            dims.extend([
                ("ffn_gate.weight", vec![e, ff]),
                ("ffn_up.weight", vec![e, ff]),
                ("ffn_down.weight", vec![ff, e]),
            ]);
        } else {
            let latent = self.latent as u64;
            let narrow = self.experts.dims(latent);
            let mut block = self.experts.dims(e);
            for ((_, dims), (_, in_latent)) in block.iter_mut().zip(narrow) {
                if dims.len() == 3 {
                    *dims = in_latent;
                }
            }
            dims.extend(block);
            dims.extend([
                ("ffn_routed_down.weight", vec![e, latent]),
                ("ffn_routed_up.weight", vec![latent, e]),
                ("ffn_routed_norm.weight", vec![latent]),
            ]);
        }
        dims
    }
}

impl Indexer {
    /// A K3.1 file's indexer keys: the head count and key width the kernels run, a rope of no
    /// values (the key is not rotated), the LayerNorm's epsilon, the top-k, and the layers that
    /// run it, which are the latent layers and no other. Every other value, or a layer list that
    /// says otherwise, is refused by the key's name.
    fn read(split: &Split, kinds: &[Kind]) -> Result<Indexer, PlacementError> {
        let exact = |suffix: &str, want: usize, why: &str| -> Result<usize, PlacementError> {
            let v = positive(split, suffix)?;
            if v == want {
                Ok(v)
            } else {
                Err(metadata(split, suffix, format!("is {v}; {why}")))
            }
        };
        let heads = exact(
            "attention.indexer.head_count",
            INDEX_HEADS,
            "the score kernel runs 32 index heads",
        )?;
        let d = exact(
            "attention.indexer.key_length",
            INDEX_DIM,
            "the key kernel's transform and the score kernel are 128 values wide",
        )?;
        let rope = meta_usize(split, "attention.indexer.rope_dimension_count")?;
        if rope != 0 {
            return Err(metadata(
                split,
                "attention.indexer.rope_dimension_count",
                format!("is {rope}; the indexer's key is not rotated (no rope values)"),
            ));
        }
        let layers = per_layer_bool(split, "attention.indexer.types", kinds.len())?;
        for (l, (&runs, &kind)) in layers.iter().zip(kinds).enumerate() {
            if runs != (kind == Kind::Mla) {
                return Err(metadata(
                    split,
                    "attention.indexer.types",
                    format!(
                        "is {} at layer {l}, a {} layer: the indexer is on every latent layer \
                         and no other",
                        u8::from(runs),
                        if kind == Kind::Mla { "latent" } else { "delta" }
                    ),
                ));
            }
        }
        Ok(Indexer {
            heads,
            d,
            top_k: positive(split, "attention.indexer.top_k")?,
            norm_eps: finite_positive(split, "attention.layer_norm_epsilon")?,
            layers,
        })
    }
}

/// Every tensor of `split` is held in a type the engine reads its stem in
/// ([`released::allowed_types`], the union of the released set, the engine's quantization and the
/// checkpoint as `prepare` writes it): the first tensor of another type, by name, with how many
/// more there are. A name that is no stem of the table is not this check's: the classifier lists
/// it.
fn types_agree(split: &Split, n_layer: usize) -> Result<(), PlacementError> {
    let mut off: Vec<(String, String)> = Vec::new();
    for (_, t) in split.iter_tensors() {
        let stem = block_stem(&t.name, n_layer).map_or(t.name.as_str(), |(_, stem)| stem);
        let Some(allowed) = released::allowed_types(stem) else {
            continue;
        };
        if !allowed.contains(&t.ty) {
            let list: Vec<String> = allowed.iter().map(ToString::to_string).collect();
            off.push((
                t.name.clone(),
                format!(
                    "{} is {} (the engine reads {stem} as {})",
                    t.name,
                    t.ty,
                    list.join(" or ")
                ),
            ));
        }
    }
    match off.first() {
        None => Ok(()),
        Some((name, _)) => Err(PlacementError::Tensor {
            name: name.clone(),
            detail: format!(
                "is of a type the Kimi K3 reader does not run the tensor in ({} such: {})",
                off.len(),
                off.iter()
                    .map(|(_, d)| d.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }),
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::{Hparams, INDEX_DIM, INDEX_HEADS, Kind, unread_keys};
    use crate::arch::synthetic::{Ty, V, header_typed};
    use gguf::Split;
    use gguf::arch::kimi_k3 as released;
    use models::KdaGate;

    /// The widths of [`keys`]: 4 layers, delta layers at 0, 1 and 3 and a latent one at 2, the
    /// first dense, a bank every 2 layers; 512 wide, 2 heads, delta heads of 128; the latent
    /// layer's ranks 256 and 512, a head of 128 + 64 query values and 128 value values; dense
    /// 768; 4 experts of 256 in a latent of 256 (not the hidden width), 2 used, 2 shared. Every
    /// first dim is a whole block of its type.
    pub(in crate::arch::kimi_k3) const E: u64 = 512;
    pub(in crate::arch::kimi_k3) const HEADS: u64 = 2;
    pub(in crate::arch::kimi_k3) const D: u64 = 128;
    pub(in crate::arch::kimi_k3) const CONV: u64 = 4;
    pub(in crate::arch::kimi_k3) const Q_LORA: u64 = 256;
    pub(in crate::arch::kimi_k3) const KV_LORA: u64 = 512;
    pub(in crate::arch::kimi_k3) const NOPE: u64 = 128;
    pub(in crate::arch::kimi_k3) const PE: u64 = 64;
    pub(in crate::arch::kimi_k3) const HEAD_V: u64 = 128;
    pub(in crate::arch::kimi_k3) const LAYERS: usize = 4;
    pub(in crate::arch::kimi_k3) const MLA_LAYERS: [usize; 1] = [2];
    pub(in crate::arch::kimi_k3) const FF: u64 = 768;
    pub(in crate::arch::kimi_k3) const N_EXP: u64 = 4;
    pub(in crate::arch::kimi_k3) const EFF: u64 = 256;
    pub(in crate::arch::kimi_k3) const SHARED: u64 = 2;
    pub(in crate::arch::kimi_k3) const LATENT: u64 = 256;
    /// The two-token vocabulary of the synthetic header.
    pub(in crate::arch::kimi_k3) const VOCAB: u64 = 2;
    /// The top-k of [`keys31`]: below the context a gate runs at, so the selection is a choice
    /// there and the context check has a side to refuse.
    pub(in crate::arch::kimi_k3) const TOP_K: u64 = 16;
    /// The index heads and key width of [`keys31`]: the kernels' own, whatever the trunk's
    /// widths.
    pub(in crate::arch::kimi_k3) const IX_HEADS: u64 = INDEX_HEADS as u64;
    pub(in crate::arch::kimi_k3) const IX_D: u64 = INDEX_DIM as u64;

    /// The keys of the released file, at this table's widths.
    pub(in crate::arch::kimi_k3) fn keys() -> Vec<(&'static str, V)> {
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
            // The file's own value, which does not describe the cache and is not read.
            ("attention.value_length", V::U32(74)),
            ("attention.key_length_mla", V::U32((NOPE + PE) as u32)),
            ("attention.value_length_mla", V::U32(HEAD_V as u32)),
            ("attention.q_lora_rank", V::U32(Q_LORA as u32)),
            ("attention.kv_lora_rank", V::U32(KV_LORA as u32)),
            ("attention.layer_norm_rms_epsilon", V::F32(1e-5)),
            ("rope.dimension_count", V::U32(PE as u32)),
            ("ssm.conv_kernel", V::U32(CONV as u32)),
            ("kda.head_dim", V::U32(D as u32)),
            ("kda.gate_lower_bound", V::F32(-5.0)),
            ("expert_count", V::U32(N_EXP as u32)),
            ("expert_used_count", V::U32(2)),
            ("expert_feed_forward_length", V::U32(EFF as u32)),
            ("expert_shared_count", V::U32(SHARED as u32)),
            ("expert_gating_func", V::U32(2)),
            ("expert_weights_scale", V::F32(1.0)),
            ("expert_weights_norm", V::Bool(true)),
            ("expert_latent_length", V::U32(LATENT as u32)),
            ("attn_res.block_size", V::U32(2)),
            ("activation.situ_beta", V::F32(4.0)),
            ("activation.situ_linear_beta", V::F32(25.0)),
        ]
    }

    /// The released file's tensor table (2,573 tensors at 93 layers), at [`keys`]' widths: the
    /// model-level four, then each layer's mixer, mixes and feed-forward block. Written from the
    /// table, apart from the reader.
    pub(in crate::arch::kimi_k3) fn tensors() -> Vec<(String, Vec<u64>)> {
        let mut t: Vec<(String, Vec<u64>)> = vec![
            ("token_embd.weight".into(), vec![E, VOCAB]),
            ("output.weight".into(), vec![E, VOCAB]),
            ("output_norm.weight".into(), vec![E]),
            ("output_res_score.weight".into(), vec![E]),
        ];
        let inner = HEADS * D;
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
                    ("ssm_f_a.weight", vec![E, D]),
                    ("ssm_f_b.weight", vec![D, inner]),
                    ("ssm_g.weight", vec![E, inner]),
                    ("ssm_beta.weight", vec![E, HEADS]),
                    ("ssm_a", vec![HEADS]),
                    ("ssm_dt.bias", vec![inner]),
                    ("ssm_norm.weight", vec![D]),
                    ("attn_output.weight", vec![inner, E]),
                ]);
            }
            layer.extend([
                ("attn_res_score.weight", vec![E]),
                ("ffn_res_score.weight", vec![E]),
                ("ffn_norm.weight", vec![E]),
            ]);
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
                    ("ffn_gate_exps.weight", vec![LATENT, EFF, N_EXP]),
                    ("ffn_up_exps.weight", vec![LATENT, EFF, N_EXP]),
                    ("ffn_down_exps.weight", vec![EFF, LATENT, N_EXP]),
                    ("ffn_routed_down.weight", vec![E, LATENT]),
                    ("ffn_routed_up.weight", vec![LATENT, E]),
                    ("ffn_routed_norm.weight", vec![LATENT]),
                    ("ffn_gate_shexp.weight", vec![E, SHARED * EFF]),
                    ("ffn_up_shexp.weight", vec![E, SHARED * EFF]),
                    ("ffn_down_shexp.weight", vec![SHARED * EFF, E]),
                ]);
            }
            t.extend(layer.into_iter().map(|(s, d)| (format!("blk.{l}.{s}"), d)));
        }
        t
    }

    /// [`keys`] with `key` replaced by `v`, or dropped when `v` is `None`.
    pub(in crate::arch::kimi_k3) fn with(
        key: &'static str,
        v: Option<V>,
    ) -> Vec<(&'static str, V)> {
        let mut kv: Vec<_> = keys().into_iter().filter(|(k, _)| *k != key).collect();
        if let Some(v) = v {
            kv.push((key, v));
        }
        kv
    }

    /// The keys of a K3.1 file at this table's widths: [`keys`] and the indexer's, on the latent
    /// layer alone.
    pub(in crate::arch::kimi_k3) fn keys31() -> Vec<(&'static str, V)> {
        let types: Vec<bool> = (0..LAYERS).map(|l| MLA_LAYERS.contains(&l)).collect();
        let mut kv = keys();
        kv.extend([
            ("attention.indexer.head_count", V::U32(IX_HEADS as u32)),
            ("attention.indexer.key_length", V::U32(IX_D as u32)),
            ("attention.indexer.top_k", V::U32(TOP_K as u32)),
            ("attention.indexer.types", V::Bools(types)),
            ("attention.indexer.rope_dimension_count", V::U32(0)),
            ("attention.layer_norm_epsilon", V::F32(1e-5)),
        ]);
        kv
    }

    /// [`keys31`] with `key` replaced by `v`, or dropped when `v` is `None`.
    pub(in crate::arch::kimi_k3) fn with31(
        key: &'static str,
        v: Option<V>,
    ) -> Vec<(&'static str, V)> {
        let mut kv: Vec<_> = keys31().into_iter().filter(|(k, _)| *k != key).collect();
        if let Some(v) = v {
            kv.push((key, v));
        }
        kv
    }

    /// The five tensors of one layer's indexer, at [`keys31`]'s widths.
    pub(in crate::arch::kimi_k3) fn indexer_tensors(l: usize) -> Vec<(String, Vec<u64>)> {
        [
            ("indexer.attn_q_b.weight", vec![Q_LORA, IX_HEADS * IX_D]),
            ("indexer.attn_k.weight", vec![E, IX_D]),
            ("indexer.proj.weight", vec![E, IX_HEADS]),
            ("indexer.k_norm.weight", vec![IX_D]),
            ("indexer.k_norm.bias", vec![IX_D]),
        ]
        .into_iter()
        .map(|(s, d)| (format!("blk.{l}.{s}"), d))
        .collect()
    }

    /// The tensors of a K3.1 file: [`tensors`] and the indexer of each latent layer.
    pub(in crate::arch::kimi_k3) fn tensors31() -> Vec<(String, Vec<u64>)> {
        let mut t = tensors();
        for l in MLA_LAYERS {
            t.extend(indexer_tensors(l));
        }
        t
    }

    /// The K3.1 tensors with `name` dropped.
    fn without31(name: &str) -> Vec<(String, Vec<u64>)> {
        tensors31().into_iter().filter(|(n, _)| n != name).collect()
    }

    /// The K3.1 tensors with `name` given `dims`, or added when the table has none.
    fn shaped31(name: &str, dims: Vec<u64>) -> Vec<(String, Vec<u64>)> {
        let mut t = without31(name);
        t.push((name.to_string(), dims));
        t
    }

    /// The type the released UD-IQ1_S file holds `name`'s stem in (the K3.1 indexer's matrices
    /// in BF16, as the engine's quantization leaves them): Q8_0 on the two narrow stems, IQ1_S
    /// on the routed stacks, F32 on a stem the table holds only in F32 and on a name it has no
    /// stem of, Q6_K on the rest. Written from the file, apart from the table.
    pub(in crate::arch::kimi_k3) fn ty_of(name: &str) -> Ty {
        let stem = name
            .strip_prefix("blk.")
            .and_then(|rest| rest.split_once('.'))
            .map_or(name, |(_, stem)| stem);
        match stem {
            "ssm_f_b.weight" | "attn_k_b.weight" => Ty::Q8_0,
            "indexer.attn_q_b.weight" | "indexer.attn_k.weight" | "indexer.proj.weight" => Ty::Bf16,
            s if s.ends_with("_exps.weight") => Ty::Iq1S,
            _ => match released::allowed_types(stem) {
                None => Ty::F32,
                Some(types) if types == [gguf::GgmlType::F32] => Ty::F32,
                Some(_) => Ty::Q6K,
            },
        }
    }

    /// `tensors` as the released set types them, `over` giving a tensor another type.
    pub(in crate::arch::kimi_k3) fn typed(
        tensors: &[(String, Vec<u64>)],
        over: &[(&str, Ty)],
    ) -> Vec<(String, Vec<u64>, Ty)> {
        tensors
            .iter()
            .map(|(n, d)| {
                let ty = over
                    .iter()
                    .find(|(name, _)| name == n)
                    .map_or_else(|| ty_of(n), |(_, ty)| *ty);
                (n.clone(), d.clone(), ty)
            })
            .collect()
    }

    /// The reader's verdict on a header of `kv` and the typed `tensors`, as the error text.
    fn read_typed(
        tag: &str,
        arch: &str,
        kv: &[(&str, V)],
        tensors: &[(String, Vec<u64>, Ty)],
    ) -> Result<Hparams, String> {
        let path = header_typed(&format!("kimi-k3-hparams-{tag}"), arch, kv, &[], tensors);
        let split = Split::open(&path).map_err(|e| e.to_string());
        let got = split.and_then(|s| Hparams::read(&s).map_err(|e| e.to_string()));
        let _ = std::fs::remove_file(&path);
        got
    }

    /// The reader's verdict on a header of `kv` and `tensors` in the released set's types.
    fn read(
        tag: &str,
        kv: &[(&str, V)],
        tensors: &[(String, Vec<u64>)],
    ) -> Result<Hparams, String> {
        read_typed(tag, "kimi-k3", kv, &typed(tensors, &[]))
    }

    /// The reader's verdict on a K3.1 header of `kv` and `tensors` in the released set's types.
    fn read31(
        tag: &str,
        kv: &[(&str, V)],
        tensors: &[(String, Vec<u64>)],
    ) -> Result<Hparams, String> {
        read_typed(tag, "kimi-k31", kv, &typed(tensors, &[]))
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
    pub(in crate::arch::kimi_k3) fn without(name: &str) -> Vec<(String, Vec<u64>)> {
        tensors().into_iter().filter(|(n, _)| n != name).collect()
    }

    /// The tensors with `name` given `dims`, or added when the table has none.
    pub(in crate::arch::kimi_k3) fn shaped(name: &str, dims: Vec<u64>) -> Vec<(String, Vec<u64>)> {
        let mut t = without(name);
        t.push((name.to_string(), dims));
        t
    }

    /// The header the table gives reads, to the values its keys hold, with the head counts
    /// splitting the layers and no default taken.
    #[test]
    fn a_small_header_reads() {
        let hp = read("small", &keys(), &tensors()).expect("the header reads");
        assert_eq!((hp.n_layer, hp.n_embd, hp.n_head), (LAYERS, 512, 2));
        assert_eq!((hp.kda.conv, hp.kda.head_dim), (4, 128));
        assert_eq!(hp.gate, KdaGate::Sigmoid { lower_bound: -5.0 });
        assert_eq!(
            (
                hp.mla.q_lora,
                hp.mla.kv_lora,
                hp.mla.head_k,
                hp.mla.head_v,
                hp.mla.pe
            ),
            (256, 512, 192, 128, 64)
        );
        assert_eq!((hp.mla.nope(), hp.mla.row()), (128, 576));
        assert_eq!(hp.rms_eps, 1e-5);
        assert_eq!((hp.situ.gate_beta, hp.situ.value_beta), (4.0, 25.0));
        assert_eq!((hp.latent, hp.attn_res_block), (256, 2));
        assert_eq!(
            (hp.n_vocab, hp.n_ctx_train, hp.dense_lead, hp.dense_ff),
            (2, 4096, 1, 768)
        );
        assert_eq!(
            (
                hp.experts.n_expert,
                hp.experts.n_used,
                hp.experts.ff,
                hp.experts.shared_ff()
            ),
            (4, 2, 256, 512)
        );
        assert_eq!(
            (hp.experts.weights_scale, hp.experts.weights_norm),
            (1.0, true)
        );
        assert_eq!((hp.inner(), hp.kda_state()), (256, [2, 128, 128]));
        let kinds: Vec<Kind> = (0..LAYERS).map(|l| hp.kind(l)).collect();
        assert_eq!(kinds, [Kind::Kda, Kind::Kda, Kind::Mla, Kind::Kda]);
        assert!(hp.dense(0) && !hp.dense(1));
        assert!(
            hp.defaults.is_empty(),
            "the file's value_length is no default: {:?}",
            hp.defaults
        );
    }

    /// A layer banks its input when its index is a multiple of the block size. Mutant: the rule
    /// off by one, or the first layer left out.
    #[test]
    fn the_bank_rule_is_every_block_size_th_layer() {
        let hp = read("banks", &keys(), &tensors()).expect("the header reads");
        let banks: Vec<bool> = (0..LAYERS).map(|l| hp.banks(l)).collect();
        assert_eq!(banks, [true, false, true, false]);
        let wide = read(
            "banks-3",
            &with("attn_res.block_size", Some(V::U32(3))),
            &tensors(),
        )
        .expect("the header reads");
        let banks: Vec<bool> = (0..LAYERS).map(|l| wide.banks(l)).collect();
        assert_eq!(banks, [true, false, false, true]);
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

    /// The routed stacks run in the latent and the router and shared expert read the hidden row:
    /// layer 1's dims put each at its width, none of them at the other's. Mutant: the latent and
    /// the hidden width exchanged in a stack, the router or the shared expert.
    #[test]
    fn the_routed_block_splits_latent_from_hidden() {
        let hp = read("split", &keys(), &tensors()).expect("the header reads");
        let dims = hp.layer_dims(1);
        let of = |stem: &str| {
            dims.iter()
                .find(|(s, _)| *s == stem)
                .unwrap_or_else(|| panic!("{stem} has no dims"))
                .1
                .clone()
        };
        assert_eq!(of("ffn_gate_exps.weight"), [LATENT, EFF, N_EXP]);
        assert_eq!(of("ffn_up_exps.weight"), [LATENT, EFF, N_EXP]);
        assert_eq!(of("ffn_down_exps.weight"), [EFF, LATENT, N_EXP]);
        assert_eq!(of("ffn_gate_inp.weight"), [E, N_EXP]);
        assert_eq!(of("ffn_gate_shexp.weight"), [E, SHARED * EFF]);
        assert_eq!(of("ffn_down_shexp.weight"), [SHARED * EFF, E]);
        assert_eq!(of("ffn_routed_down.weight"), [E, LATENT]);
        assert_eq!(of("ffn_routed_up.weight"), [LATENT, E]);
        assert_eq!(of("ffn_routed_norm.weight"), [LATENT]);
        assert_eq!(of("attn_res_score.weight"), [E]);
        assert_eq!(of("ssm_g.weight"), [E, HEADS * D]);
    }

    /// Every refusal names its key or tensor. Mutants: a check dropped, or the wrong name given.
    #[test]
    fn the_reader_refuses_by_name() {
        let cases: Vec<Case> = vec![
            (
                "no-head-dim".into(),
                with("kda.head_dim", None),
                tensors(),
                "kda.head_dim",
            ),
            (
                "no-conv".into(),
                with("ssm.conv_kernel", None),
                tensors(),
                "ssm.conv_kernel",
            ),
            (
                "no-latent".into(),
                with("expert_latent_length", None),
                tensors(),
                "expert_latent_length",
            ),
            (
                "zero-latent".into(),
                with("expert_latent_length", Some(V::U32(0))),
                tensors(),
                "expert_latent_length",
            ),
            (
                "no-block".into(),
                with("attn_res.block_size", None),
                tensors(),
                "attn_res.block_size",
            ),
            (
                "zero-block".into(),
                with("attn_res.block_size", Some(V::U32(0))),
                tensors(),
                "attn_res.block_size",
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
                "no-pe".into(),
                with("rope.dimension_count", None),
                tensors(),
                "rope.dimension_count",
            ),
            (
                "pe-fills-head".into(),
                with("rope.dimension_count", Some(V::U32(192))),
                tensors(),
                "attention.key_length_mla",
            ),
            (
                "no-row-width".into(),
                with("attention.key_length", None),
                tensors(),
                "attention.key_length",
            ),
            (
                "row-width".into(),
                with("attention.key_length", Some(V::U32(640))),
                tensors(),
                "attention.key_length",
            ),
            (
                "softmax".into(),
                with("expert_gating_func", Some(V::U32(1))),
                tensors(),
                "expert_gating_func",
            ),
            (
                "two-groups".into(),
                with("expert_group_used_count", Some(V::U32(2))),
                tensors(),
                "expert_group_used_count",
            ),
            (
                "lead-all".into(),
                with("leading_dense_block_count", Some(V::U32(LAYERS as u32))),
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
                with("attention.head_count_kv", Some(V::I32s(vec![0, 0, 2, 0]))),
                tensors(),
                "2 at layer 2",
            ),
            (
                "all-delta".into(),
                with("attention.head_count_kv", Some(V::I32s(vec![0; LAYERS]))),
                tensors(),
                "4 delta layers of 4",
            ),
            (
                "all-latent".into(),
                with("attention.head_count_kv", Some(V::I32s(vec![1; LAYERS]))),
                tensors(),
                "0 delta layers of 4",
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
                "no-output-gate".into(),
                keys(),
                without("blk.1.ssm_g.weight"),
                "blk.1.ssm_g.weight",
            ),
            (
                "no-mixer-score".into(),
                keys(),
                without("blk.1.attn_res_score.weight"),
                "blk.1.attn_res_score.weight",
            ),
            (
                "no-ffn-score".into(),
                keys(),
                without("blk.3.ffn_res_score.weight"),
                "blk.3.ffn_res_score.weight",
            ),
            (
                "no-final-score".into(),
                keys(),
                without("output_res_score.weight"),
                "output_res_score.weight",
            ),
            (
                "no-routed-norm".into(),
                keys(),
                without("blk.3.ffn_routed_norm.weight"),
                "blk.3.ffn_routed_norm.weight",
            ),
            (
                "no-routed-up".into(),
                keys(),
                without("blk.1.ffn_routed_up.weight"),
                "blk.1.ffn_routed_up.weight",
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
                shaped("blk.2.ssm_g.weight", vec![E, HEADS * D]),
                "blk.2.ssm_g.weight",
            ),
            (
                "dense-with-router".into(),
                keys(),
                shaped("blk.0.ffn_gate_inp.weight", vec![E, N_EXP]),
                "blk.0.ffn_gate_inp.weight",
            ),
            (
                "dense-with-latent".into(),
                keys(),
                shaped("blk.0.ffn_routed_down.weight", vec![E, LATENT]),
                "blk.0.ffn_routed_down.weight",
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
                shaped("blk.1.ssm_f_b.weight", vec![2 * D, HEADS * D]),
                "blk.1.ssm_f_b.weight",
            ),
            (
                "latent-row".into(),
                keys(),
                shaped("blk.2.attn_kv_a_mqa.weight", vec![E, KV_LORA]),
                "blk.2.attn_kv_a_mqa.weight",
            ),
            (
                "k-b".into(),
                keys(),
                shaped("blk.2.attn_k_b.weight", vec![NOPE, KV_LORA, 1]),
                "blk.2.attn_k_b.weight",
            ),
            (
                "stack-in-hidden".into(),
                keys(),
                shaped("blk.3.ffn_gate_exps.weight", vec![E, EFF, N_EXP]),
                "blk.3.ffn_gate_exps.weight",
            ),
            (
                "down-stack-in-hidden".into(),
                keys(),
                shaped("blk.1.ffn_down_exps.weight", vec![EFF, E, N_EXP]),
                "blk.1.ffn_down_exps.weight",
            ),
            (
                "router-in-latent".into(),
                keys(),
                shaped("blk.3.ffn_gate_inp.weight", vec![LATENT, N_EXP]),
                "blk.3.ffn_gate_inp.weight",
            ),
            (
                "shared-in-latent".into(),
                keys(),
                shaped("blk.3.ffn_gate_shexp.weight", vec![LATENT, SHARED * EFF]),
                "blk.3.ffn_gate_shexp.weight",
            ),
            (
                "shared-width".into(),
                keys(),
                shaped("blk.3.ffn_down_shexp.weight", vec![EFF, E]),
                "blk.3.ffn_down_shexp.weight",
            ),
            (
                "routed-down-swapped".into(),
                keys(),
                shaped("blk.3.ffn_routed_down.weight", vec![LATENT, E]),
                "blk.3.ffn_routed_down.weight",
            ),
            (
                "routed-norm-hidden".into(),
                keys(),
                shaped("blk.1.ffn_routed_norm.weight", vec![E]),
                "blk.1.ffn_routed_norm.weight",
            ),
            (
                "score-width".into(),
                keys(),
                shaped("blk.1.attn_res_score.weight", vec![LATENT]),
                "blk.1.attn_res_score.weight",
            ),
            (
                "final-score-width".into(),
                keys(),
                shaped("output_res_score.weight", vec![LATENT]),
                "output_res_score.weight",
            ),
            (
                "output-gate-width".into(),
                keys(),
                shaped("blk.1.ssm_g.weight", vec![E, HEADS * D + 1]),
                "blk.1.ssm_g.weight",
            ),
        ];
        let wrong: Vec<String> = cases
            .iter()
            .filter_map(|(tag, kv, t, want)| wrong_verdict(tag, kv, t, want))
            .collect();
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    /// A tensor of a type the table does not hold its stem in is refused by name, and every type
    /// the released file holds reads: the routed stacks in IQ1_S on every layer and in IQ2_XXS on
    /// the layers its recipe gives them. Mutant: the check dropped, or the table cut to one type.
    #[test]
    fn a_type_outside_the_table_is_refused_by_name() {
        let all = tensors();
        let mixed = typed(
            &all,
            &[
                ("blk.1.ffn_gate_exps.weight", Ty::Iq2Xxs),
                ("blk.1.ffn_up_exps.weight", Ty::Iq2Xxs),
                ("blk.3.ffn_down_exps.weight", Ty::Iq2Xxs),
            ],
        );
        read_typed("mixed", "kimi-k3", &keys(), &mixed).expect("the union's mixes read");
        for (tag, name, ty) in [
            ("expert-q6k", "blk.3.ffn_gate_exps.weight", Ty::Q6K),
            ("embd-f16", "token_embd.weight", Ty::F16),
            ("k-b-f32", "blk.2.attn_k_b.weight", Ty::F32),
            ("score-f16", "blk.1.attn_res_score.weight", Ty::F16),
            ("router-q8", "blk.3.ffn_gate_inp.weight", Ty::Q8_0),
            ("latent-down-iq1", "blk.1.ffn_routed_down.weight", Ty::Iq1S),
            // The narrow stems run Q8_0 and BF16 alone: their first dim is 128 at this width, which
            // no 256-wide block fits, so the quantized trunk's types are not written there (the
            // gguf table's test holds that).
            ("decay-rank-f16", "blk.1.ssm_f_b.weight", Ty::F16),
            ("head-q4k", "output.weight", Ty::Q4K),
            ("embd-q8", "token_embd.weight", Ty::Q8_0),
            ("projection-f16", "blk.1.ssm_g.weight", Ty::F16),
        ] {
            let got = read_typed(tag, "kimi-k3", &keys(), &typed(&all, &[(name, ty)]));
            match got {
                Ok(_) => panic!("{tag}: read"),
                Err(e) => assert!(
                    e.contains(name) && e.contains("the engine reads"),
                    "{tag}: {e}"
                ),
            }
        }
    }

    /// The engine's own quantization (trunk Q4_K, IQ1_S routed stacks) and the checkpoint's
    /// types (BF16 projections, the stems that are F32 in both) read through the same table as
    /// the released file. The checkpoint's MXFP4 stacks have no synthetic writer here: the
    /// `prepare` gate in `bloomery-model-hf` writes and opens them.
    #[test]
    fn the_quantization_and_the_checkpoint_read_with_the_released_file() {
        let released_ty = |n: &str| ty_of(n);
        let quant: Vec<_> = tensors31()
            .into_iter()
            .map(|(n, d)| {
                let ty = match released_ty(&n) {
                    Ty::Q6K if !n.ends_with("attn_v_b.weight") && !n.ends_with("output.weight") => {
                        Ty::Q4K
                    }
                    other => other,
                };
                (n, d, ty)
            })
            .collect();
        read_typed("quant", "kimi-k31", &keys31(), &quant).expect("the quantization reads");
        let checkpoint: Vec<_> = tensors31()
            .into_iter()
            .map(|(n, d)| {
                let ty = match released_ty(&n) {
                    Ty::Q6K | Ty::Q8_0 => Ty::Bf16,
                    other => other,
                };
                (n, d, ty)
            })
            .collect();
        read_typed("checkpoint", "kimi-k31", &keys31(), &checkpoint)
            .expect("the checkpoint's projections read");
        let released_file = typed(&tensors(), &[]);
        read_typed("released", "kimi-k3", &keys(), &released_file)
            .expect("the released file reads");
    }

    /// The K3.1 table's header reads to the indexer its keys give, on the latent layer alone, and
    /// the layer's dims hold the five indexer tensors there and nowhere else. A K3 file has none.
    #[test]
    fn a_k31_header_reads_its_indexer() {
        let hp = read31("k31", &keys31(), &tensors31()).expect("the K3.1 header reads");
        assert_eq!(hp.arch, "kimi-k31");
        let ix = hp.indexer.as_ref().expect("the indexer");
        assert_eq!(
            (ix.heads, ix.d, ix.top_k, ix.norm_eps),
            (32, 128, TOP_K as usize, 1e-5)
        );
        assert_eq!(ix.layers, [false, false, true, false]);
        let dsa: Vec<bool> = (0..LAYERS).map(|l| hp.dsa(l)).collect();
        assert_eq!(dsa, [false, false, true, false]);
        let of = |l: usize, stem: &str| {
            hp.layer_dims(l)
                .into_iter()
                .find(|(s, _)| *s == stem)
                .map(|(_, d)| d)
        };
        assert_eq!(
            of(2, "indexer.attn_q_b.weight"),
            Some(vec![Q_LORA, IX_HEADS * IX_D])
        );
        assert_eq!(of(2, "indexer.attn_k.weight"), Some(vec![E, IX_D]));
        assert_eq!(of(2, "indexer.proj.weight"), Some(vec![E, IX_HEADS]));
        assert_eq!(of(2, "indexer.k_norm.weight"), Some(vec![IX_D]));
        assert_eq!(of(2, "indexer.k_norm.bias"), Some(vec![IX_D]));
        for l in [0, 1, 3] {
            assert_eq!(of(l, "indexer.proj.weight"), None, "layer {l}");
        }
        let k3 = read("k3", &keys(), &tensors()).expect("the K3 header reads");
        assert!(
            k3.indexer.is_none() && !k3.dsa(2),
            "a K3 file has no indexer"
        );
    }

    /// One K3.1 refusal case: a tag, the header's keys, the tensor shapes and the name the
    /// refusal must hold.
    type Case31 = (
        &'static str,
        Vec<(&'static str, V)>,
        Vec<(String, Vec<u64>)>,
        &'static str,
    );

    /// Every K3.1 key and tensor check refuses by name: a head count, key width or rope the
    /// kernels do not run, a missing or zero top-k and epsilon, a layer list that puts the indexer
    /// off a latent layer or on a delta one, and the indexer's tensors missing, misshapen or on
    /// a layer that has none. Mutant: a check dropped, or the wrong name given.
    #[test]
    fn the_k31_reader_refuses_by_name() {
        let ix = |l: usize, stem: &str| format!("blk.{l}.indexer.{stem}.weight");
        let cases: Vec<Case31> = vec![
            (
                "no-heads",
                with31("attention.indexer.head_count", None),
                tensors31(),
                "attention.indexer.head_count",
            ),
            (
                "heads-64",
                with31("attention.indexer.head_count", Some(V::U32(64))),
                tensors31(),
                "attention.indexer.head_count",
            ),
            (
                "key-length-64",
                with31("attention.indexer.key_length", Some(V::U32(64))),
                tensors31(),
                "attention.indexer.key_length",
            ),
            (
                "no-key-length",
                with31("attention.indexer.key_length", None),
                tensors31(),
                "attention.indexer.key_length",
            ),
            (
                "rope-64",
                with31("attention.indexer.rope_dimension_count", Some(V::U32(64))),
                tensors31(),
                "attention.indexer.rope_dimension_count",
            ),
            (
                "no-rope",
                with31("attention.indexer.rope_dimension_count", None),
                tensors31(),
                "attention.indexer.rope_dimension_count",
            ),
            (
                "no-top-k",
                with31("attention.indexer.top_k", None),
                tensors31(),
                "attention.indexer.top_k",
            ),
            (
                "zero-top-k",
                with31("attention.indexer.top_k", Some(V::U32(0))),
                tensors31(),
                "attention.indexer.top_k",
            ),
            (
                "no-types",
                with31("attention.indexer.types", None),
                tensors31(),
                "attention.indexer.types",
            ),
            (
                "types-off-the-latent-layer",
                with31(
                    "attention.indexer.types",
                    Some(V::Bools(vec![false; LAYERS])),
                ),
                tensors31(),
                "attention.indexer.types",
            ),
            (
                "types-on-a-delta-layer",
                with31(
                    "attention.indexer.types",
                    Some(V::Bools(vec![false, true, true, false])),
                ),
                tensors31(),
                "attention.indexer.types",
            ),
            (
                "types-short",
                with31("attention.indexer.types", Some(V::Bools(vec![false, true]))),
                tensors31(),
                "attention.indexer.types",
            ),
            (
                "no-epsilon",
                with31("attention.layer_norm_epsilon", None),
                tensors31(),
                "attention.layer_norm_epsilon",
            ),
            (
                "zero-epsilon",
                with31("attention.layer_norm_epsilon", Some(V::F32(0.0))),
                tensors31(),
                "attention.layer_norm_epsilon",
            ),
            (
                "q-b-heads",
                keys31(),
                shaped31(&ix(2, "attn_q_b"), vec![Q_LORA, 64 * IX_D]),
                "blk.2.indexer.attn_q_b.weight",
            ),
            (
                "q-b-in-hidden",
                keys31(),
                shaped31(&ix(2, "attn_q_b"), vec![E, IX_HEADS * IX_D]),
                "blk.2.indexer.attn_q_b.weight",
            ),
            (
                "k-width",
                keys31(),
                shaped31(&ix(2, "attn_k"), vec![E, 64]),
                "blk.2.indexer.attn_k.weight",
            ),
            (
                "proj-heads",
                keys31(),
                shaped31(&ix(2, "proj"), vec![E, 16]),
                "blk.2.indexer.proj.weight",
            ),
            (
                "k-norm-width",
                keys31(),
                shaped31(&ix(2, "k_norm"), vec![64]),
                "blk.2.indexer.k_norm.weight",
            ),
            (
                "k-norm-bias-width",
                keys31(),
                shaped31("blk.2.indexer.k_norm.bias", vec![64]),
                "blk.2.indexer.k_norm.bias",
            ),
            (
                "no-q-b",
                keys31(),
                without31(&ix(2, "attn_q_b")),
                "blk.2.indexer.attn_q_b.weight",
            ),
            (
                "no-k",
                keys31(),
                without31(&ix(2, "attn_k")),
                "blk.2.indexer.attn_k.weight",
            ),
            (
                "no-proj",
                keys31(),
                without31(&ix(2, "proj")),
                "blk.2.indexer.proj.weight",
            ),
            (
                "no-k-norm",
                keys31(),
                without31(&ix(2, "k_norm")),
                "blk.2.indexer.k_norm.weight",
            ),
            (
                "no-k-norm-bias",
                keys31(),
                without31("blk.2.indexer.k_norm.bias"),
                "blk.2.indexer.k_norm.bias",
            ),
            (
                "indexer-on-a-delta-layer",
                keys31(),
                shaped31(&ix(1, "proj"), vec![E, IX_HEADS]),
                "blk.1.indexer.proj.weight",
            ),
            (
                "bias-on-a-delta-layer",
                keys31(),
                shaped31("blk.3.indexer.k_norm.bias", vec![IX_D]),
                "blk.3.indexer.k_norm.bias",
            ),
            (
                "k3-tensors-under-the-k31-word",
                keys31(),
                tensors(),
                "blk.2.indexer.attn_q_b.weight",
            ),
        ];
        let wrong: Vec<String> = cases
            .iter()
            .filter_map(|(tag, kv, t, want)| match read31(tag, kv, t) {
                Ok(_) => Some(format!("{tag}: read; expected {want:?}")),
                Err(e) if e.contains(want) => None,
                Err(e) => Some(format!("{tag}: want {want:?} in: {e}")),
            })
            .collect();
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    /// A K3 file with an indexer tensor is refused by that tensor's name: the indexer is K3.1's.
    #[test]
    fn an_indexer_tensor_in_a_k3_file_is_refused_by_name() {
        let mut t = tensors();
        t.extend(indexer_tensors(2));
        let e = read("k3-indexer", &keys(), &t).expect_err("a K3 file has no indexer");
        assert!(e.contains("blk.2.indexer.attn_q_b.weight"), "{e}");
    }

    /// A type the indexer's stems have no kernel for is refused by name: the matrices run BF16
    /// or Q8_0, the key's LayerNorm F32 alone. Mutant: the indexer stems left out of the table.
    #[test]
    fn an_indexer_tensor_of_another_type_is_refused_by_name() {
        let all = tensors31();
        for (tag, name, ty) in [
            ("proj-q4k", "blk.2.indexer.proj.weight", Ty::Q4K),
            ("k-f16", "blk.2.indexer.attn_k.weight", Ty::F16),
            ("q-b-iq1", "blk.2.indexer.attn_q_b.weight", Ty::Iq1S),
            ("norm-bf16", "blk.2.indexer.k_norm.weight", Ty::Bf16),
            ("bias-f16", "blk.2.indexer.k_norm.bias", Ty::F16),
        ] {
            let got = read_typed(tag, "kimi-k31", &keys31(), &typed(&all, &[(name, ty)]));
            match got {
                Ok(_) => panic!("{tag}: read"),
                Err(e) => assert!(
                    e.contains(name) && e.contains("the engine reads"),
                    "{tag}: {e}"
                ),
            }
        }
        read_typed(
            "indexer-q8",
            "kimi-k31",
            &keys31(),
            &typed(&all, &[("blk.2.indexer.proj.weight", Ty::Q8_0)]),
        )
        .expect("a Q8_0 indexer matrix reads");
    }

    /// A K3.1 file runs every context: at its indexer's top-k the selection is every position, past
    /// it the latent layers attend the rows the indexer picks, so a context of `top_k + 1` and one
    /// of 262,144 positions (256k) run as a K3 file's do. Mutant: the bound past `top_k` put back.
    #[test]
    fn a_k31_context_past_top_k_runs() {
        let hp = read31("ctx", &keys31(), &tensors31()).expect("the K3.1 header reads");
        for ctx in [1, TOP_K, TOP_K + 1, 262_144] {
            hp.check_ctx(ctx)
                .unwrap_or_else(|e| panic!("a context of {ctx} runs: {e}"));
        }
        let k3 = read("ctx-k3", &keys(), &tensors()).expect("the K3 header reads");
        k3.check_ctx(1 << 20).expect("a K3 file has no bound");
    }

    /// The reader lists, and does not refuse, the keys of a K3.1 file it does not read: the
    /// indexer's are read.
    #[test]
    fn a_k31_file_reads_its_indexer_keys() {
        let mut kv = keys31();
        kv.push(("rope.freq_base", V::F32(10_000.0)));
        let path = header_typed(
            "kimi-k31-hparams-unread",
            "kimi-k31",
            &kv,
            &[],
            &typed(&tensors31(), &[]),
        );
        let split = Split::open(&path).expect("opens");
        let mut unread = unread_keys(&split);
        let _ = std::fs::remove_file(&path);
        unread.sort();
        assert_eq!(
            unread,
            vec![
                "kimi-k31.attention.value_length".to_string(),
                "kimi-k31.rope.freq_base".to_string()
            ]
        );
    }

    /// Another architecture's file is refused by `general.architecture`, whatever its keys hold.
    #[test]
    fn another_architecture_is_refused_by_name() {
        for arch in ["deepseek41", "glm5next", "raon1", "kimi-linear"] {
            let got = read_typed("arch", arch, &keys(), &typed(&tensors(), &[]));
            match got {
                Ok(_) => panic!("{arch}: read"),
                Err(e) => assert!(
                    e.contains(gguf::GENERAL_ARCHITECTURE) && e.contains(arch),
                    "{arch}: {e}"
                ),
            }
        }
    }

    /// The reader lists, and does not refuse, a key it does not read: the file's stale
    /// `attention.value_length` and its `rope.freq_base`, which no layer turns by.
    #[test]
    fn an_unread_key_is_listed() {
        let mut kv = keys();
        kv.push(("rope.freq_base", V::F32(10_000.0)));
        let path = header_typed(
            "kimi-k3-hparams-unread",
            "kimi-k3",
            &kv,
            &[],
            &typed(&tensors(), &[]),
        );
        let split = Split::open(&path).expect("opens");
        let mut unread = unread_keys(&split);
        let _ = std::fs::remove_file(&path);
        unread.sort();
        assert_eq!(
            unread,
            vec![
                "kimi-k3.attention.value_length".to_string(),
                "kimi-k3.rope.freq_base".to_string()
            ]
        );
    }
}
