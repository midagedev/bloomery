//! Every `minimax-m3.*` key the description reads, and the tensors each layer is held to. A key
//! the loader requires is required here, and a value this reader does not run is refused by the
//! key's name. Which layers are dense and which sparse (an indexer, routed experts and a shared
//! expert) is `leading_dense_block_count`; the tensors a layer carries must be exactly its kind's
//! ([`Hparams::read`]). The dialect's authority is llama.cpp's `src/models/minimax-m3.cpp`.

use gguf::Split;

use super::roles::{ATTENTION, DENSE, INDEXER, ROUTED, required};
use crate::arch::{
    dims_agree, expert_counts, finite_positive, meta_bool, meta_usize, metadata, mul, n_vocab,
    one_expert_group, optional_usize, plain_rope, positive, refuse_tensor_set, sigmoid_gating,
    unread_keys_of,
};
use crate::placement::PlacementError;

/// `swigluoai`'s sigmoid slope (`alpha`), the constant llama.cpp's graph passes for every
/// activation of the architecture (`src/llama-graph.cpp`, the dense and routed branches).
pub const SWIGLU_ALPHA: f32 = 1.702;

/// `swigluoai`'s clamp (`limit`): the gate's upper bound and the up path's two bounds.
pub const SWIGLU_LIMIT: f32 = 7.0;

/// What a layer is, from `leading_dense_block_count`: the first layers are dense, the rest sparse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Attends every position; a dense feed-forward block.
    Dense,
    /// Attends the blocks its indexer picks; routed experts beside a shared one.
    Sparse,
}

/// The hyperparameters of one minimax-m3 file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// `block_count`: every layer is a trunk layer, the file carries no next-token block.
    pub n_layer: usize,
    /// `embedding_length`.
    pub n_embd: usize,
    /// `attention.head_count`: query heads.
    pub n_head: usize,
    /// `attention.head_count_kv`: key and value heads, the same on every layer.
    pub n_head_kv: usize,
    /// `attention.key_length`, which `attention.value_length` equals: one head's query, key and
    /// value width.
    pub head_dim: usize,
    /// The partial NeoX rope of q, k and the indexer's query and key.
    pub rope: Rope,
    /// `attention.layer_norm_rms_epsilon`: every RMS norm's, the per-head ones included.
    pub rms_eps: f32,
    /// `vocab_size`, else the token list's length.
    pub n_vocab: usize,
    /// `context_length`.
    pub n_ctx_train: usize,
    /// `leading_dense_block_count`: the layers before the first sparse one.
    pub dense_lead: usize,
    /// `feed_forward_length`, the dense layers' width; `None` on a file with no dense layer.
    pub dense_ff: Option<usize>,
    /// The sparse layers' router and experts.
    pub experts: Experts,
    /// The sparse layers' block selection.
    pub msa: Msa,
    /// The keys this read took a default for: the key, the value, the line.
    pub defaults: Vec<String>,
}

/// The rope every attention layer applies: NeoX pairs (`i`, `i + dims/2`) of the first `dims`
/// values of a head, frequency scale 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rope {
    /// `rope.dimension_count`: rotated values, even and within the head.
    pub dims: usize,
    /// `rope.freq_base`.
    pub base: f32,
}

/// The sparse layers' sigmoid noaux_tc router and `swigluoai` experts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Experts {
    /// `expert_count`.
    pub n_expert: usize,
    /// `expert_used_count`.
    pub n_used: usize,
    /// `expert_feed_forward_length`: one routed expert's width, and one shared expert's.
    pub ff: usize,
    /// `expert_shared_count`: shared experts, run as one block of `ff · n_shared` values.
    pub n_shared: usize,
    /// `expert_weights_scale`: the factor on the kept weights.
    pub weights_scale: f32,
    /// `expert_weights_norm`: the kept weights are renormalized to sum to one.
    pub weights_norm: bool,
}

impl Experts {
    /// The shared block's width.
    #[must_use]
    pub fn shared_ff(&self) -> usize {
        self.ff * self.n_shared
    }
}

/// The sparse layers' selection of position blocks (MiniMax Sparse Attention).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Msa {
    /// `attention.indexer.head_count`: index heads, one for each key head.
    pub heads: usize,
    /// `attention.indexer.key_length`: values per index head and per index key.
    pub d: usize,
    /// `attention.indexer.top_k`: blocks a query attends, its own among them.
    pub top_k: usize,
    /// `attention.indexer.block_size`: positions a block holds.
    pub block: usize,
    /// `attention.indexer.local_blocks`: blocks, the query's own first, kept whatever they score.
    pub local: usize,
}

/// Every `minimax-m3.` key [`Hparams::read`] reads, by suffix.
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
    "attention.layer_norm_rms_epsilon",
    "attention.indexer.head_count",
    "attention.indexer.key_length",
    "attention.indexer.top_k",
    "attention.indexer.block_size",
    "attention.indexer.local_blocks",
    "rope.dimension_count",
    "rope.freq_base",
    "rope.scaling.type",
    "rope.scaling.factor",
    "expert_count",
    "expert_used_count",
    "expert_feed_forward_length",
    "expert_shared_count",
    "expert_gating_func",
    "expert_weights_scale",
    "expert_weights_norm",
    "expert_group_count",
    "expert_group_used_count",
];

/// The architecture keys and `general.sampling.*` keys of `split` that [`Hparams::read`] does not
/// read: listed, not refused.
#[must_use]
pub fn unread_keys(split: &Split) -> Vec<String> {
    unread_keys_of(split, READ)
}

impl Hparams {
    /// Every value from `split`'s headers; a missing required key, a value this reader does not
    /// run, a next-token tensor, or a layer whose tensors disagree with its kind or the keys is
    /// an error naming the key or the tensor.
    pub fn read(split: &Split) -> Result<Hparams, PlacementError> {
        let mut defaults = Vec::new();
        let n_layer = positive(split, "block_count")?;
        refuse_mtp(split, n_layer)?;
        let n_embd = positive(split, "embedding_length")?;
        let n_head = positive(split, "attention.head_count")?;
        let n_head_kv = positive(split, "attention.head_count_kv")?;
        if !n_head.is_multiple_of(n_head_kv) {
            return Err(metadata(
                split,
                "attention.head_count_kv",
                format!(
                    "is {n_head_kv} for {n_head} query heads; grouped-query attention needs a \
                     whole number of query heads per key head"
                ),
            ));
        }
        let head_dim = positive(split, "attention.key_length")?;
        let value = positive(split, "attention.value_length")?;
        if value != head_dim {
            return Err(metadata(
                split,
                "attention.value_length",
                format!("is {value}, attention.key_length {head_dim}; one width serves Q, K and V"),
            ));
        }
        let rope = Rope::read(split, head_dim)?;
        let dense_lead = match optional_usize(split, "leading_dense_block_count")? {
            Some(n) => n,
            None => {
                defaults.push("leading_dense_block_count = 0 (absent)".to_string());
                0
            }
        };
        if dense_lead >= n_layer {
            return Err(metadata(
                split,
                "leading_dense_block_count",
                format!("is {dense_lead} for {n_layer} layers; at least one layer must be sparse"),
            ));
        }
        let dense_ff = (dense_lead > 0)
            .then(|| positive(split, "feed_forward_length"))
            .transpose()?;
        let hp = Hparams {
            n_layer,
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
            rope,
            rms_eps: finite_positive(split, "attention.layer_norm_rms_epsilon")?,
            n_vocab: n_vocab(split)?,
            n_ctx_train: positive(split, "context_length")?,
            dense_lead,
            dense_ff,
            experts: Experts::read(split, &mut defaults)?,
            msa: Msa::read(split, n_head_kv, &rope)?,
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
        if l < self.dense_lead {
            Kind::Dense
        } else {
            Kind::Sparse
        }
    }

    /// Query heads per key and value head.
    #[must_use]
    pub fn group(&self) -> usize {
        self.n_head / self.n_head_kv
    }

    /// Every layer carries its kind's required tensors, an indexer exactly where it is sparse, and
    /// nothing of the other kind's: the first missing or misplaced name, with how many more there
    /// are.
    fn tensors_agree(&self, split: &Split) -> Result<(), PlacementError> {
        let has = |l: usize, stem: &str| split.find(&format!("blk.{l}.{stem}")).is_some();
        let mut missing = Vec::new();
        let mut misplaced = Vec::new();
        for l in 0..self.n_layer {
            let sparse = self.kind(l) == Kind::Sparse;
            let (own, other) = if sparse {
                (ROUTED, DENSE)
            } else {
                (DENSE, ROUTED)
            };
            let mut need: Vec<&str> = required(ATTENTION).collect();
            need.extend(required(own));
            if sparse {
                need.extend(required(INDEXER));
            }
            missing.extend(
                need.iter()
                    .filter(|s| !has(l, s))
                    .map(|s| format!("blk.{l}.{s}")),
            );
            let mut banned: Vec<&str> = other
                .iter()
                .map(|s| s.name)
                .filter(|s| !need.contains(s))
                .collect();
            if !sparse {
                banned.extend(INDEXER.iter().map(|s| s.name));
            }
            misplaced.extend(
                banned
                    .iter()
                    .filter(|s| has(l, s))
                    .map(|s| format!("blk.{l}.{s}")),
            );
        }
        refuse_tensor_set(
            &missing,
            &misplaced,
            "the layers' kinds need",
            "its layer's kind carries none",
        )
    }

    /// Layer `l`'s tensors by stem and the dims llama.cpp creates them with, ggml order.
    #[must_use]
    pub fn layer_dims(&self, l: usize) -> Vec<(&'static str, Vec<u64>)> {
        let (e, h, kv, k) = (
            self.n_embd as u64,
            self.n_head as u64,
            self.n_head_kv as u64,
            self.head_dim as u64,
        );
        let mut dims = vec![
            ("attn_norm.weight", vec![e]),
            ("attn_q.weight", vec![e, h * k]),
            ("attn_k.weight", vec![e, kv * k]),
            ("attn_v.weight", vec![e, kv * k]),
            ("attn_q_norm.weight", vec![k]),
            ("attn_k_norm.weight", vec![k]),
            ("attn_output.weight", vec![h * k, e]),
            ("ffn_norm.weight", vec![e]),
        ];
        match self.kind(l) {
            Kind::Dense => {
                let ff = self.dense_ff.unwrap_or(0) as u64;
                dims.extend([
                    ("ffn_gate.weight", vec![e, ff]),
                    ("ffn_up.weight", vec![e, ff]),
                    ("ffn_down.weight", vec![ff, e]),
                ]);
            }
            Kind::Sparse => {
                let x = &self.experts;
                let (n, eff, shared) = (x.n_expert as u64, x.ff as u64, x.shared_ff() as u64);
                let m = &self.msa;
                let (ih, id) = (m.heads as u64, m.d as u64);
                dims.extend([
                    ("ffn_gate_inp.weight", vec![e, n]),
                    ("exp_probs_b.bias", vec![n]),
                    ("ffn_gate_exps.weight", vec![e, eff, n]),
                    ("ffn_up_exps.weight", vec![e, eff, n]),
                    ("ffn_down_exps.weight", vec![eff, e, n]),
                    ("ffn_gate_shexp.weight", vec![e, shared]),
                    ("ffn_up_shexp.weight", vec![e, shared]),
                    ("ffn_down_shexp.weight", vec![shared, e]),
                    ("indexer.q_proj.weight", vec![e, ih * id]),
                    ("indexer.k_proj.weight", vec![e, id]),
                    ("indexer.q_norm.weight", vec![id]),
                    ("indexer.k_norm.weight", vec![id]),
                ]);
            }
        }
        dims
    }
}

/// A file with a multi-token-prediction layer is a new reader decision, not a layer to ignore: a
/// non-zero `nextn_predict_layers`, a `nextn.*` tensor, or a block past `block_count` is refused
/// by name.
fn refuse_mtp(split: &Split, n_layer: usize) -> Result<(), PlacementError> {
    let key = "nextn_predict_layers";
    if let Some(n) = optional_usize(split, key)?
        && n != 0
    {
        return Err(metadata(
            split,
            key,
            format!(
                "is {n}; the released weights carry no next-token layer, and a file with one \
                 needs a reader decision"
            ),
        ));
    }
    let past: Vec<String> = split
        .iter_tensors()
        .filter(|(_, t)| {
            t.name
                .strip_prefix("blk.")
                .and_then(|rest| rest.split_once('.'))
                .is_some_and(|(block, stem)| {
                    stem.starts_with("nextn.") || block.parse::<usize>().is_ok_and(|b| b >= n_layer)
                })
        })
        .map(|(_, t)| t.name.clone())
        .collect();
    match past.first() {
        None => Ok(()),
        Some(name) => Err(PlacementError::Tensor {
            name: name.clone(),
            detail: format!(
                "is a next-token tensor ({} such: {}); the released weights carry none, and a \
                 file with them needs a reader decision",
                past.len(),
                past.join(", ")
            ),
        }),
    }
}

impl Rope {
    fn read(split: &Split, head_dim: usize) -> Result<Rope, PlacementError> {
        let dims = positive(split, "rope.dimension_count")?;
        if !dims.is_multiple_of(2) || dims > head_dim {
            return Err(metadata(
                split,
                "rope.dimension_count",
                format!(
                    "is {dims}, for a head of {head_dim}; NeoX rotates whole pairs inside the head"
                ),
            ));
        }
        plain_rope(split)?;
        Ok(Rope {
            dims,
            base: finite_positive(split, "rope.freq_base")?,
        })
    }
}

impl Experts {
    fn read(split: &Split, defaults: &mut Vec<String>) -> Result<Experts, PlacementError> {
        let (n_expert, n_used) = expert_counts(split)?;
        let ff = positive(split, "expert_feed_forward_length")?;
        let n_shared = positive(split, "expert_shared_count")?;
        mul(split, "expert_shared_count", ff, n_shared)?;
        sigmoid_gating(split, None, defaults)?;
        one_expert_group(split, "the minimax-m3 router")?;
        Ok(Experts {
            n_expert,
            n_used,
            ff,
            n_shared,
            weights_scale: finite_positive(split, "expert_weights_scale")?,
            weights_norm: meta_bool(split, "expert_weights_norm")?,
        })
    }
}

impl Msa {
    fn read(split: &Split, n_head_kv: usize, rope: &Rope) -> Result<Msa, PlacementError> {
        let heads = positive(split, "attention.indexer.head_count")?;
        if heads != n_head_kv {
            return Err(metadata(
                split,
                "attention.indexer.head_count",
                format!(
                    "is {heads}, attention.head_count_kv {n_head_kv}; each key head selects on \
                     its own, so there is one index head for each"
                ),
            ));
        }
        let d = positive(split, "attention.indexer.key_length")?;
        if d < rope.dims {
            return Err(metadata(
                split,
                "attention.indexer.key_length",
                format!(
                    "is {d}, under rope.dimension_count {}; the index query and key rotate \
                     the same leading values",
                    rope.dims
                ),
            ));
        }
        let top_k = positive(split, "attention.indexer.top_k")?;
        let block = positive(split, "attention.indexer.block_size")?;
        let local = meta_usize(split, "attention.indexer.local_blocks")?;
        if local == 0 || local > top_k {
            return Err(metadata(
                split,
                "attention.indexer.local_blocks",
                format!(
                    "is {local} for top_k {top_k}; the query's own block is always kept inside \
                     the top_k, and no more than top_k blocks are"
                ),
            ));
        }
        Ok(Msa {
            heads,
            d,
            top_k,
            block,
            local,
        })
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::{Hparams, Kind, unread_keys};
    use crate::arch::synthetic::{V, header_shaped};
    use gguf::Split;

    /// The widths of [`keys`]: 5 layers, 2 dense then 3 sparse; 64 wide, 8 query heads and 2 key
    /// heads of 16, rotating 8 of them; dense 48; 8 experts of 24, 2 used, 1 shared; 2 index heads
    /// of 16, blocks of 4 keeping 3.
    pub(in crate::arch::minimaxm3) const E: u64 = 64;
    pub(in crate::arch::minimaxm3) const HEADS: u64 = 8;
    pub(in crate::arch::minimaxm3) const KV: u64 = 2;
    pub(in crate::arch::minimaxm3) const K: u64 = 16;
    pub(in crate::arch::minimaxm3) const LAYERS: usize = 5;
    pub(in crate::arch::minimaxm3) const LEAD: usize = 2;
    pub(in crate::arch::minimaxm3) const FF: u64 = 48;
    pub(in crate::arch::minimaxm3) const N_EXP: u64 = 8;
    pub(in crate::arch::minimaxm3) const EFF: u64 = 24;
    pub(in crate::arch::minimaxm3) const IDX_HEADS: u64 = 2;
    pub(in crate::arch::minimaxm3) const IDX_D: u64 = 16;
    /// The two-token vocabulary of the synthetic header.
    pub(in crate::arch::minimaxm3) const VOCAB: u64 = 2;

    pub(in crate::arch::minimaxm3) fn keys() -> Vec<(&'static str, V)> {
        vec![
            ("block_count", V::U32(LAYERS as u32)),
            ("context_length", V::U32(4096)),
            ("embedding_length", V::U32(E as u32)),
            ("feed_forward_length", V::U32(FF as u32)),
            ("leading_dense_block_count", V::U32(LEAD as u32)),
            ("attention.head_count", V::U32(HEADS as u32)),
            ("attention.head_count_kv", V::U32(KV as u32)),
            ("attention.key_length", V::U32(K as u32)),
            ("attention.value_length", V::U32(K as u32)),
            ("attention.layer_norm_rms_epsilon", V::F32(1e-6)),
            ("rope.freq_base", V::F32(5_000_000.0)),
            ("rope.dimension_count", V::U32(8)),
            ("expert_count", V::U32(N_EXP as u32)),
            ("expert_used_count", V::U32(2)),
            ("expert_feed_forward_length", V::U32(EFF as u32)),
            ("expert_shared_count", V::U32(1)),
            ("expert_gating_func", V::U32(2)),
            ("expert_weights_scale", V::F32(2.0)),
            ("expert_weights_norm", V::Bool(true)),
            ("attention.indexer.head_count", V::U32(IDX_HEADS as u32)),
            ("attention.indexer.key_length", V::U32(IDX_D as u32)),
            ("attention.indexer.top_k", V::U32(3)),
            ("attention.indexer.block_size", V::U32(4)),
            ("attention.indexer.local_blocks", V::U32(1)),
        ]
    }

    /// The released file's tensor table (1,176 tensors at 60 layers), at [`keys`]' widths: the
    /// three model-level tensors, then each layer's attention and norm tensors and its kind's
    /// block, the sparse layers' indexer last. Written from the table, apart from the reader.
    pub(in crate::arch::minimaxm3) fn tensors() -> Vec<(String, Vec<u64>)> {
        tensors_lead(LEAD)
    }

    /// [`tensors`] with the first `lead` layers dense.
    fn tensors_lead(lead: usize) -> Vec<(String, Vec<u64>)> {
        let mut t: Vec<(String, Vec<u64>)> = vec![
            ("token_embd.weight".into(), vec![E, VOCAB]),
            ("output.weight".into(), vec![E, VOCAB]),
            ("output_norm.weight".into(), vec![E]),
        ];
        for l in 0..LAYERS {
            let mut layer: Vec<(&str, Vec<u64>)> = vec![
                ("attn_norm.weight", vec![E]),
                ("attn_q.weight", vec![E, HEADS * K]),
                ("attn_k.weight", vec![E, KV * K]),
                ("attn_v.weight", vec![E, KV * K]),
                ("attn_q_norm.weight", vec![K]),
                ("attn_k_norm.weight", vec![K]),
                ("attn_output.weight", vec![HEADS * K, E]),
                ("ffn_norm.weight", vec![E]),
            ];
            if l < lead {
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
                    ("indexer.q_proj.weight", vec![E, IDX_HEADS * IDX_D]),
                    ("indexer.k_proj.weight", vec![E, IDX_D]),
                    ("indexer.q_norm.weight", vec![IDX_D]),
                    ("indexer.k_norm.weight", vec![IDX_D]),
                ]);
            }
            t.extend(layer.into_iter().map(|(s, d)| (format!("blk.{l}.{s}"), d)));
        }
        t
    }

    /// [`keys`] with `key` replaced by `v`, or dropped when `v` is `None`.
    pub(in crate::arch::minimaxm3) fn with(
        key: &'static str,
        v: Option<V>,
    ) -> Vec<(&'static str, V)> {
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
        let path = header_shaped(
            &format!("mm3-hparams-{tag}"),
            "minimax-m3",
            kv,
            &[],
            tensors,
        );
        let split = Split::open(&path).map_err(|e| e.to_string());
        let got = split.and_then(|s| Hparams::read(&s).map_err(|e| e.to_string()));
        let _ = std::fs::remove_file(&path);
        got
    }

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

    fn refused(tag: &str, kv: &[(&str, V)], tensors: &[(String, Vec<u64>)], want: &str) {
        if let Some(wrong) = wrong_verdict(tag, kv, tensors, want) {
            panic!("{wrong}");
        }
    }

    /// A refusal to check: the header's tag, keys and tensors, and the text its error holds.
    type Case = (
        String,
        Vec<(&'static str, V)>,
        Vec<(String, Vec<u64>)>,
        String,
    );

    /// Every verdict wrong among `cases` (tag, keys, tensors, wanted text), asserted at once, so a
    /// mutant shows every case it turns.
    fn assert_all_refused(cases: Vec<Case>) {
        let wrong: Vec<String> = cases
            .iter()
            .filter_map(|(tag, kv, t, want)| wrong_verdict(tag, kv, t, want))
            .collect();
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    /// The header the table gives reads, to the values its keys hold, with the dense lead
    /// splitting the layers and no default taken.
    #[test]
    fn a_small_header_reads() {
        let hp = read("small", &keys(), &tensors()).expect("the header reads");
        assert_eq!(
            (hp.n_layer, hp.n_embd, hp.n_head, hp.n_head_kv, hp.head_dim),
            (LAYERS, 64, 8, 2, 16)
        );
        assert_eq!((hp.rope.dims, hp.rope.base), (8, 5_000_000.0));
        assert_eq!(
            (hp.n_vocab, hp.n_ctx_train, hp.dense_ff),
            (2, 4096, Some(48))
        );
        assert_eq!(hp.group(), 4);
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
            (2.0, true)
        );
        assert_eq!(
            (
                hp.msa.heads,
                hp.msa.d,
                hp.msa.top_k,
                hp.msa.block,
                hp.msa.local
            ),
            (2, 16, 3, 4, 1)
        );
        let kinds: Vec<Kind> = (0..LAYERS).map(|l| hp.kind(l)).collect();
        assert_eq!(
            kinds,
            [
                Kind::Dense,
                Kind::Dense,
                Kind::Sparse,
                Kind::Sparse,
                Kind::Sparse
            ]
        );
        assert!(hp.defaults.is_empty(), "{:?}", hp.defaults);
    }

    /// The released file's hyperparameters, at its real keys: 60 layers, the first 3 dense.
    pub(in crate::arch::minimaxm3) fn real() -> Hparams {
        // Each caller reads a header of its own: tests run in parallel and a header's path is its tag.
        static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut hp =
            read(&format!("real-keys-{call}"), &keys(), &tensors()).expect("the header reads");
        hp.n_layer = 60;
        hp.n_embd = 6144;
        hp.n_head = 64;
        hp.n_head_kv = 4;
        hp.head_dim = 128;
        hp.rope.dims = 64;
        hp.dense_lead = 3;
        hp.dense_ff = Some(12288);
        hp.experts.n_expert = 128;
        hp.experts.n_used = 4;
        hp.experts.ff = 3072;
        hp.msa.heads = 4;
        hp.msa.d = 128;
        hp.msa.top_k = 16;
        hp.msa.block = 128;
        hp
    }

    /// The table the reader holds a file to has the released file's 1,176 tensors and its listed
    /// dims. Mutant: a sparse layer without its indexer, or the shared expert counted twice.
    #[test]
    fn the_released_files_table_has_1176_tensors() {
        let hp = real();
        let model_level = 3;
        let layered: usize = (0..60).map(|l| hp.layer_dims(l).len()).sum();
        assert_eq!(model_level + layered, 1176);
        let dims = |l: usize, stem: &str| {
            hp.layer_dims(l)
                .into_iter()
                .find(|(s, _)| *s == stem)
                .map(|(_, d)| d)
        };
        assert_eq!(dims(0, "attn_q.weight"), Some(vec![6144, 8192]));
        assert_eq!(dims(0, "attn_k.weight"), Some(vec![6144, 512]));
        assert_eq!(dims(59, "attn_output.weight"), Some(vec![8192, 6144]));
        assert_eq!(dims(2, "ffn_down.weight"), Some(vec![12288, 6144]));
        assert_eq!(dims(2, "indexer.q_proj.weight"), None);
        assert_eq!(dims(3, "ffn_gate.weight"), None);
        assert_eq!(dims(3, "ffn_gate_exps.weight"), Some(vec![6144, 3072, 128]));
        assert_eq!(dims(3, "ffn_down_exps.weight"), Some(vec![3072, 6144, 128]));
        assert_eq!(dims(3, "ffn_gate_shexp.weight"), Some(vec![6144, 3072]));
        assert_eq!(dims(3, "indexer.q_proj.weight"), Some(vec![6144, 512]));
        assert_eq!(dims(3, "indexer.k_proj.weight"), Some(vec![6144, 128]));
    }

    /// An absent dense lead is none, recorded as a default: every layer is then sparse, and a file
    /// whose first layers are dense is refused for the router they lack. Mutant: the absent key
    /// read as the file's dense count.
    #[test]
    fn an_absent_dense_lead_is_none_and_recorded() {
        let kv = with("leading_dense_block_count", None);
        let hp = read("lead-absent", &kv, &tensors_lead(0)).expect("every layer sparse reads");
        assert_eq!((hp.dense_lead, hp.dense_ff), (0, None));
        assert_eq!(hp.kind(0), Kind::Sparse);
        assert_eq!(hp.defaults, ["leading_dense_block_count = 0 (absent)"]);
        refused(
            "lead-absent-dense",
            &kv,
            &tensors(),
            "blk.0.ffn_gate_inp.weight: is not in the file",
        );
    }

    /// Each key the reader requires is named when absent.
    #[test]
    fn a_missing_key_is_named() {
        let cases: Vec<_> = [
            "attention.head_count_kv",
            "attention.key_length",
            "attention.indexer.head_count",
            "attention.indexer.key_length",
            "attention.indexer.top_k",
            "attention.indexer.block_size",
            "attention.indexer.local_blocks",
            "rope.dimension_count",
            "rope.freq_base",
            "expert_shared_count",
            "expert_gating_func",
            "expert_weights_scale",
            "expert_weights_norm",
            "feed_forward_length",
        ]
        .map(|key| {
            (
                format!("missing-{key}"),
                with(key, None),
                tensors(),
                format!("minimax-m3.{key}"),
            )
        })
        .into_iter()
        .collect();
        assert_all_refused(cases);
    }

    /// A value the reader does not run is refused with the key's name and the value.
    #[test]
    fn unsupported_values_are_refused_by_key_and_value() {
        let cases = [
            (
                "kv-divides",
                with("attention.head_count_kv", Some(V::U32(3))),
                "attention.head_count_kv: is 3 for 8 query heads",
            ),
            (
                "value-width",
                with("attention.value_length", Some(V::U32(8))),
                "attention.value_length: is 8, attention.key_length 16",
            ),
            (
                "rope-odd",
                with("rope.dimension_count", Some(V::U32(7))),
                "rope.dimension_count: is 7, for a head of 16",
            ),
            (
                "rope-wide",
                with("rope.dimension_count", Some(V::U32(32))),
                "rope.dimension_count: is 32, for a head of 16",
            ),
            (
                "rope-yarn",
                with("rope.scaling.type", Some(V::Str("yarn"))),
                "rope.scaling.type: is \"yarn\"",
            ),
            (
                "gating",
                with("expert_gating_func", Some(V::U32(1))),
                "expert_gating_func: is 1",
            ),
            (
                "groups",
                with("expert_group_count", Some(V::U32(2))),
                "expert_group_count: is 2",
            ),
            (
                "used",
                with("expert_used_count", Some(V::U32(9))),
                "expert_used_count: is 9 of expert_count 8",
            ),
            (
                "scale",
                with("expert_weights_scale", Some(V::F32(0.0))),
                "expert_weights_scale: is 0",
            ),
            (
                "lead-all",
                with("leading_dense_block_count", Some(V::U32(5))),
                "leading_dense_block_count: is 5 for 5 layers",
            ),
            (
                "idx-heads",
                with("attention.indexer.head_count", Some(V::U32(4))),
                "attention.indexer.head_count: is 4, attention.head_count_kv 2",
            ),
            (
                "idx-narrow",
                with("attention.indexer.key_length", Some(V::U32(4))),
                "attention.indexer.key_length: is 4, under rope.dimension_count 8",
            ),
            (
                "local-zero",
                with("attention.indexer.local_blocks", Some(V::U32(0))),
                "attention.indexer.local_blocks: is 0 for top_k 3",
            ),
            (
                "local-over",
                with("attention.indexer.local_blocks", Some(V::U32(4))),
                "attention.indexer.local_blocks: is 4 for top_k 3",
            ),
            (
                "block-zero",
                with("attention.indexer.block_size", Some(V::U32(0))),
                "attention.indexer.block_size: is 0",
            ),
        ];
        assert_all_refused(
            cases
                .into_iter()
                .map(|(tag, kv, want)| (tag.to_string(), kv, tensors(), want.to_string()))
                .collect(),
        );
    }

    /// A sparse layer needs its indexer, a dense layer carries none, and neither carries the other
    /// kind's feed-forward block: the tensor set must be exactly the kind's. Mutant: the indexer
    /// stems dropped from the sparse layers' required set, or from the dense layers' banned set.
    #[test]
    fn a_layers_tensors_must_be_its_kinds() {
        let without = |name: &str| -> Vec<(String, Vec<u64>)> {
            tensors().into_iter().filter(|(n, _)| n != name).collect()
        };
        let plus = |name: &str, dims: Vec<u64>| -> Vec<(String, Vec<u64>)> {
            let mut t = tensors();
            t.push((name.to_string(), dims));
            t
        };
        let missing = "is not in the file";
        let misplaced = "is in the file, and its layer's kind carries none";
        let cases = [
            (
                "no-indexer",
                without("blk.3.indexer.k_proj.weight"),
                "blk.3.indexer.k_proj.weight",
                missing,
            ),
            (
                "no-router",
                without("blk.4.ffn_gate_inp.weight"),
                "blk.4.ffn_gate_inp.weight",
                missing,
            ),
            (
                "no-shexp",
                without("blk.2.ffn_up_shexp.weight"),
                "blk.2.ffn_up_shexp.weight",
                missing,
            ),
            (
                "no-dense",
                without("blk.1.ffn_down.weight"),
                "blk.1.ffn_down.weight",
                missing,
            ),
            (
                "dense-indexer",
                plus("blk.1.indexer.q_norm.weight", vec![IDX_D]),
                "blk.1.indexer.q_norm.weight",
                misplaced,
            ),
            (
                "dense-routed",
                plus("blk.0.ffn_gate_inp.weight", vec![E, N_EXP]),
                "blk.0.ffn_gate_inp.weight",
                misplaced,
            ),
            (
                "sparse-dense",
                plus("blk.3.ffn_up.weight", vec![E, FF]),
                "blk.3.ffn_up.weight",
                misplaced,
            ),
        ];
        assert_all_refused(
            cases
                .into_iter()
                .map(|(tag, t, name, why)| (tag.to_string(), keys(), t, format!("{name}: {why}")))
                .collect(),
        );
    }

    /// A tensor of other dims than the keys give it is refused by name, whichever it is. Mutant:
    /// the dims check skipped, or one tensor's formula off by one.
    #[test]
    fn a_tensor_of_other_dims_is_refused_by_name() {
        let cases = [
            ("blk.3.indexer.k_proj.weight", vec![E, IDX_D + 1]),
            ("blk.3.indexer.q_proj.weight", vec![E, IDX_D]),
            ("blk.2.attn_q.weight", vec![E, K]),
            ("blk.4.ffn_down_exps.weight", vec![E, EFF, N_EXP]),
            ("blk.0.ffn_gate.weight", vec![FF, E]),
            ("token_embd.weight", vec![E + 1, VOCAB]),
        ];
        assert_all_refused(
            cases
                .into_iter()
                .map(|(name, dims)| {
                    let t: Vec<_> = tensors()
                        .into_iter()
                        .map(|(n, d)| if n == name { (n, dims.clone()) } else { (n, d) })
                        .collect();
                    (name.to_string(), keys(), t, format!("{name} {dims:?}"))
                })
                .collect(),
        );
    }

    /// A file with a next-token layer is refused by name, whichever way it shows: the count, a
    /// `nextn.*` tensor, or a block past `block_count`. Mutant: the count's zero test loosened, or
    /// the stem test dropped.
    #[test]
    fn a_file_with_mtp_is_refused_by_name() {
        let with_extra = |name: &str, dims: Vec<u64>| {
            let mut t = tensors();
            t.push((name.to_string(), dims));
            t
        };
        assert_all_refused(vec![
            (
                "mtp-key".to_string(),
                with("nextn_predict_layers", Some(V::U32(1))),
                tensors(),
                "nextn_predict_layers: is 1".to_string(),
            ),
            (
                "mtp-stem".to_string(),
                keys(),
                with_extra("blk.4.nextn.eh_proj.weight", vec![2 * E, E]),
                "blk.4.nextn.eh_proj.weight: is a next-token tensor".to_string(),
            ),
            (
                "mtp-block".to_string(),
                keys(),
                with_extra("blk.5.attn_q.weight", vec![E, HEADS * K]),
                "blk.5.attn_q.weight: is a next-token tensor".to_string(),
            ),
        ]);
        // A count of 0 declares none.
        read(
            "mtp-zero",
            &with("nextn_predict_layers", Some(V::U32(0))),
            &tensors(),
        )
        .expect("a declared 0 is no layer");
    }

    /// Every `minimax-m3.` key the reader holds is listed as read, and a key it does not is listed
    /// as unread.
    #[test]
    fn every_key_of_the_header_is_read_or_listed() {
        let open = |kv: &[(&str, V)]| {
            let path = header_shaped("mm3-unread", "minimax-m3", kv, &[], &tensors());
            let split = Split::open(&path).expect("the synthetic header opens");
            let unread = unread_keys(&split);
            let _ = std::fs::remove_file(&path);
            unread
        };
        assert!(open(&keys()).is_empty());
        let mut kv = keys();
        kv.push(("attention.sliding_window", V::U32(8)));
        assert_eq!(open(&kv), ["minimax-m3.attention.sliding_window"]);
    }
}
