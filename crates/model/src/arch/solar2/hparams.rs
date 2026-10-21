//! Every `solar-open2.*` key the description reads, each layer's kind from
//! `attention.head_count_kv`, and the tensors the file is held to. A key the loader requires is
//! required here, and a value this reader does not run is refused by the key's name. A layer is a
//! KDA layer when its key-head count is 0 and a NoPE GQA layer of that many key and value heads
//! otherwise; the tensors a layer carries must be exactly its kind's ([`Hparams::read`]).

use gguf::Split;

use super::roles::stems_of;
use crate::arch::hybrid::{Attn, layers_agree, no_nextn};
use crate::arch::noaux::{self, Experts};
use crate::arch::{
    dims_agree, finite_positive, kda, metadata, n_vocab, optional_usize, per_layer_count, positive,
    unread_keys_of,
};
use crate::placement::PlacementError;

/// What a layer mixes with, from `attention.head_count_kv`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// 0 key heads: a KDA delta-rule layer.
    Kda,
    /// Key heads: a NoPE GQA layer with an output gate projection.
    Gqa,
}

/// The hyperparameters of one solar-open2 file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// `block_count`: every layer is a trunk layer, the file carries no next-token block.
    pub n_layer: usize,
    /// `embedding_length`.
    pub n_embd: usize,
    /// `attention.head_count`: the GQA layers' query heads and the KDA layers' key and value heads.
    pub n_head: usize,
    /// `attention.key_length`, which `attention.value_length` equals: one GQA head's query, key
    /// and value width.
    pub head_dim: usize,
    /// `attention.head_count_kv`, per layer: a GQA layer's key and value heads, 0 on a KDA layer.
    pub kv_heads: Vec<usize>,
    /// `ssm.conv_kernel` and `kda.head_dim`: the KDA layers' conv taps and head width.
    pub kda: kda::Keys,
    /// `attention.layer_norm_rms_epsilon`: every RMS norm's.
    pub rms_eps: f32,
    /// `vocab_size`, else the token list's length.
    pub n_vocab: usize,
    /// `context_length`.
    pub n_ctx_train: usize,
    /// The layers' router and experts.
    pub experts: Experts,
    /// The keys this read took a default for: the key, the value, the line.
    pub defaults: Vec<String>,
}

/// Every `solar-open2.` key [`Hparams::read`] reads but the routed block's ([`noaux::KEYS`]), by
/// suffix. The KDA keys are [`kda::read`]'s.
const READ: &[&str] = &[
    "block_count",
    "nextn_predict_layers",
    "context_length",
    "embedding_length",
    "vocab_size",
    "leading_dense_block_count",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.layer_norm_rms_epsilon",
    "ssm.conv_kernel",
    "ssm.state_size",
    "ssm.group_count",
    "kda.head_dim",
];

/// The architecture keys and `general.sampling.*` keys of `split` that [`Hparams::read`] does not
/// read: listed, not refused. The file's `feed_forward_length` (no layer is dense) and
/// `rope.freq_base` (no layer rotates) are among them.
#[must_use]
pub fn unread_keys(split: &Split) -> Vec<String> {
    unread_keys_of(split, &[READ, noaux::KEYS].concat())
}

impl Hparams {
    /// Every value from `split`'s headers; a missing required key, a value this reader does not
    /// run, or a layer whose tensors disagree with its kind or the keys is an error naming the
    /// key or the tensor.
    pub fn read(split: &Split) -> Result<Hparams, PlacementError> {
        let mut defaults = Vec::new();
        let n_layer = positive(split, "block_count")?;
        no_nextn(split)?;
        let n_embd = positive(split, "embedding_length")?;
        let n_head = positive(split, "attention.head_count")?;
        let head_dim = positive(split, "attention.key_length")?;
        let value = positive(split, "attention.value_length")?;
        if value != head_dim {
            return Err(metadata(
                split,
                "attention.value_length",
                format!("is {value}, attention.key_length {head_dim}; one width serves Q, K and V"),
            ));
        }
        let kv_heads = per_layer_count(split, "attention.head_count_kv", n_layer)?;
        let attn = Attn {
            n_head,
            key_len: head_dim,
            value_len: head_dim,
        };
        for (l, &kv) in kv_heads.iter().enumerate().filter(|(_, kv)| **kv != 0) {
            attn.check_layer(split, l, kv)?;
        }
        let dense_lead = match optional_usize(split, "leading_dense_block_count")? {
            Some(n) => n,
            None => {
                defaults.push("leading_dense_block_count = 0 (absent)".to_string());
                0
            }
        };
        if dense_lead != 0 {
            return Err(metadata(
                split,
                "leading_dense_block_count",
                format!(
                    "is {dense_lead}; every layer of this reader routes its feed-forward block"
                ),
            ));
        }
        let hp = Hparams {
            n_layer,
            n_embd,
            n_head,
            head_dim,
            kv_heads,
            kda: kda::read(split, n_head, &mut defaults)?,
            rms_eps: finite_positive(split, "attention.layer_norm_rms_epsilon")?,
            n_vocab: n_vocab(split)?,
            n_ctx_train: positive(split, "context_length")?,
            experts: Experts::read(split, "the solar-open2 router", &mut defaults)?,
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
        if self.kv_heads[l] == 0 {
            Kind::Kda
        } else {
            Kind::Gqa
        }
    }

    /// The model-level three and every layer's kind's required tensors are in the file, and
    /// nothing of the other kind's: the first missing or misplaced name, with how many more
    /// there are.
    fn tensors_agree(&self, split: &Split) -> Result<(), PlacementError> {
        let kinds: Vec<Kind> = (0..self.n_layer).map(|l| self.kind(l)).collect();
        layers_agree(split, &kinds, stems_of, &[Kind::Kda, Kind::Gqa])
    }

    /// Layer `l`'s tensors by stem and the dims llama.cpp creates them with, ggml order.
    #[must_use]
    pub fn layer_dims(&self, l: usize) -> Vec<(&'static str, Vec<u64>)> {
        let e = self.n_embd as u64;
        let mut dims = match self.kind(l) {
            Kind::Kda => kda::dims(self.n_embd, self.n_head, self.kda.head_dim, self.kda.conv)
                .into_iter()
                // The common table holds `ssm_a` as [heads], ik's [heads, 1] with the trailing 1
                // dropped; llama.cpp's converter writes [1, heads], and this check is exact.
                .map(|(stem, d)| match stem {
                    "ssm_a" => (stem, vec![1, self.n_head as u64]),
                    _ => (stem, d),
                })
                .collect(),
            Kind::Gqa => {
                let attn = Attn {
                    n_head: self.n_head,
                    key_len: self.head_dim,
                    value_len: self.head_dim,
                };
                let mut dims = vec![("attn_norm.weight", vec![e])];
                dims.extend(attn.dims(self.kv_heads[l], e));
                dims.push((
                    "attn_gate.weight",
                    vec![e, (self.n_head * self.head_dim) as u64],
                ));
                dims
            }
        };
        dims.push(("ffn_norm.weight", vec![e]));
        dims.extend(self.experts.dims(e));
        dims
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::{Hparams, Kind, unread_keys};
    use crate::arch::synthetic::{V, verdict};
    use gguf::Split;

    /// The widths of [`keys`]: 8 layers, a GQA layer at 0 and 4 and KDA between; 64 wide, 4 heads
    /// of 16, 2 key heads on the GQA layers; 8 experts of 24, 2 used, 1 shared.
    pub(in crate::arch::solar2) const E: u64 = 64;
    pub(in crate::arch::solar2) const HEADS: u64 = 4;
    pub(in crate::arch::solar2) const K: u64 = 16;
    pub(in crate::arch::solar2) const KV: u64 = 2;
    pub(in crate::arch::solar2) const CONV: u64 = 4;
    pub(in crate::arch::solar2) const N_EXP: u64 = 8;
    pub(in crate::arch::solar2) const EFF: u64 = 24;
    pub(in crate::arch::solar2) const GQA_LAYERS: [usize; 2] = [0, 4];
    pub(in crate::arch::solar2) const LAYERS: usize = 8;
    /// The two-token vocabulary of the synthetic header.
    pub(in crate::arch::solar2) const VOCAB: u64 = 2;

    /// The keys of the released file, at this table's widths (`feed_forward_length` and
    /// `rope.freq_base` among them, which no layer reads).
    pub(in crate::arch::solar2) fn keys() -> Vec<(&'static str, V)> {
        let kv: Vec<i32> = (0..LAYERS)
            .map(|l| {
                if GQA_LAYERS.contains(&l) {
                    KV as i32
                } else {
                    0
                }
            })
            .collect();
        vec![
            ("block_count", V::U32(LAYERS as u32)),
            ("context_length", V::U32(4096)),
            ("embedding_length", V::U32(E as u32)),
            ("feed_forward_length", V::U32(160)),
            ("leading_dense_block_count", V::U32(0)),
            ("attention.head_count", V::U32(HEADS as u32)),
            ("attention.head_count_kv", V::I32s(kv)),
            ("attention.key_length", V::U32(K as u32)),
            ("attention.value_length", V::U32(K as u32)),
            ("attention.layer_norm_rms_epsilon", V::F32(1e-5)),
            ("rope.freq_base", V::F32(10_000.0)),
            ("kda.head_dim", V::U32(K as u32)),
            ("ssm.conv_kernel", V::U32(CONV as u32)),
            ("expert_count", V::U32(N_EXP as u32)),
            ("expert_used_count", V::U32(2)),
            ("expert_feed_forward_length", V::U32(EFF as u32)),
            ("expert_shared_count", V::U32(1)),
            ("expert_gating_func", V::U32(2)),
            ("expert_weights_scale", V::F32(1.0)),
            ("expert_weights_norm", V::Bool(true)),
        ]
    }

    /// The released file's tensor table (1,083 tensors at 48 layers), at [`keys`]' widths: the
    /// three model-level tensors, then each layer's mixer, norm and routed block. Written from the
    /// table, apart from the reader.
    pub(in crate::arch::solar2) fn tensors() -> Vec<(String, Vec<u64>)> {
        let inner = HEADS * K;
        let mut t: Vec<(String, Vec<u64>)> = vec![
            ("token_embd.weight".into(), vec![E, VOCAB]),
            ("output.weight".into(), vec![E, VOCAB]),
            ("output_norm.weight".into(), vec![E]),
        ];
        for l in 0..LAYERS {
            let mut layer: Vec<(&str, Vec<u64>)> = vec![
                ("attn_norm.weight", vec![E]),
                ("attn_q.weight", vec![E, inner]),
                ("attn_output.weight", vec![inner, E]),
            ];
            if GQA_LAYERS.contains(&l) {
                layer.extend([
                    ("attn_k.weight", vec![E, KV * K]),
                    ("attn_v.weight", vec![E, KV * K]),
                    ("attn_gate.weight", vec![E, inner]),
                ]);
            } else {
                layer.extend([
                    ("attn_k.weight", vec![E, inner]),
                    ("attn_v.weight", vec![E, inner]),
                    ("ssm_conv1d_q.weight", vec![CONV, 1, inner]),
                    ("ssm_conv1d_k.weight", vec![CONV, 1, inner]),
                    ("ssm_conv1d_v.weight", vec![CONV, 1, inner]),
                    ("ssm_f_a.weight", vec![E, K]),
                    ("ssm_f_b.weight", vec![K, inner]),
                    ("ssm_g_a.weight", vec![E, K]),
                    ("ssm_g_b.weight", vec![K, inner]),
                    ("ssm_beta.weight", vec![E, HEADS]),
                    ("ssm_a", vec![1, HEADS]),
                    ("ssm_dt.bias", vec![inner]),
                    ("ssm_norm.weight", vec![K]),
                ]);
            }
            layer.extend([
                ("ffn_norm.weight", vec![E]),
                ("ffn_gate_inp.weight", vec![E, N_EXP]),
                ("exp_probs_b.bias", vec![N_EXP]),
                ("ffn_gate_exps.weight", vec![E, EFF, N_EXP]),
                ("ffn_up_exps.weight", vec![E, EFF, N_EXP]),
                ("ffn_down_exps.weight", vec![EFF, E, N_EXP]),
                ("ffn_gate_shexp.weight", vec![E, EFF]),
                ("ffn_up_shexp.weight", vec![E, EFF]),
                ("ffn_down_shexp.weight", vec![EFF, E]),
            ]);
            t.extend(layer.into_iter().map(|(s, d)| (format!("blk.{l}.{s}"), d)));
        }
        t
    }

    /// [`keys`] with `key` replaced by `v`, or dropped when `v` is `None`.
    pub(in crate::arch::solar2) fn with(key: &'static str, v: Option<V>) -> Vec<(&'static str, V)> {
        let mut kv: Vec<_> = keys().into_iter().filter(|(k, _)| *k != key).collect();
        if let Some(v) = v {
            kv.push((key, v));
        }
        kv
    }

    /// [`tensors`] without `name`, and with `name`'s dims replaced when `dims` is given.
    fn tensors_with(name: &str, dims: Option<Vec<u64>>) -> Vec<(String, Vec<u64>)> {
        let mut t: Vec<_> = tensors().into_iter().filter(|(n, _)| n != name).collect();
        if let Some(d) = dims {
            t.push((name.to_string(), d));
        }
        t
    }

    fn read(
        tag: &str,
        kv: &[(&str, V)],
        tensors: &[(String, Vec<u64>)],
    ) -> Result<Hparams, String> {
        verdict(
            &format!("so2-hparams-{tag}"),
            "solar-open2",
            kv,
            &[],
            tensors,
            Hparams::read,
        )
    }

    /// A refusal to check: the header's tag, keys and tensors, and the text its error holds.
    type Case = (
        String,
        Vec<(&'static str, V)>,
        Vec<(String, Vec<u64>)>,
        String,
    );

    /// Every verdict wrong among `cases`, asserted at once, so a mutant shows every case it turns.
    fn assert_all_refused(cases: Vec<Case>) {
        let wrong: Vec<String> = cases
            .iter()
            .filter_map(|(tag, kv, t, want)| match read(tag, kv, t) {
                Ok(_) => Some(format!("{tag}: read; expected {want:?}")),
                Err(e) if e.contains(want.as_str()) => None,
                Err(e) => Some(format!("{tag}: want {want:?} in: {e}")),
            })
            .collect();
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    /// The header the table gives reads, to the values its keys hold: the GQA layers are those
    /// with key heads, the others KDA, and no default is taken.
    #[test]
    fn a_small_header_reads() {
        let hp = read("small", &keys(), &tensors()).expect("the header reads");
        assert_eq!(
            (hp.n_layer, hp.n_embd, hp.n_head, hp.head_dim),
            (LAYERS, 64, 4, 16)
        );
        assert_eq!((hp.kda.conv, hp.kda.head_dim), (4, 16));
        assert_eq!((hp.n_vocab, hp.n_ctx_train), (2, 4096));
        assert_eq!(hp.rms_eps, 1e-5);
        for l in 0..LAYERS {
            let want = if GQA_LAYERS.contains(&l) {
                (Kind::Gqa, KV as usize)
            } else {
                (Kind::Kda, 0)
            };
            assert_eq!((hp.kind(l), hp.kv_heads[l]), want, "layer {l}");
        }
        let x = hp.experts;
        assert_eq!(
            (x.n_expert, x.n_used, x.ff, x.shared_ff(), x.weights_scale),
            (8, 2, 24, 24, 1.0)
        );
        assert!(x.weights_norm);
        assert!(hp.defaults.is_empty(), "{:?}", hp.defaults);
    }

    /// The keys no layer reads are listed, not refused: `feed_forward_length` (no layer is dense)
    /// and `rope.freq_base` (no layer rotates).
    #[test]
    fn the_unread_keys_are_listed() {
        let path = crate::arch::synthetic::header_shaped(
            "so2-hparams-unread",
            "solar-open2",
            &keys(),
            &[("general.sampling.temp", V::F32(0.8))],
            &tensors(),
        );
        let split = Split::open(&path).expect("the synthetic header opens");
        let unread = unread_keys(&split);
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            unread,
            [
                "solar-open2.feed_forward_length",
                "solar-open2.rope.freq_base",
                "general.sampling.temp"
            ]
        );
    }

    /// Each key the reader requires is named when absent; `kda.head_dim` falls back to
    /// `ssm.state_size`, which is then the key named.
    #[test]
    fn a_missing_key_is_named() {
        let cases: Vec<_> = [
            ("block_count", "block_count"),
            ("embedding_length", "embedding_length"),
            ("context_length", "context_length"),
            ("attention.head_count", "attention.head_count"),
            ("attention.head_count_kv", "attention.head_count_kv"),
            ("attention.key_length", "attention.key_length"),
            ("attention.value_length", "attention.value_length"),
            (
                "attention.layer_norm_rms_epsilon",
                "attention.layer_norm_rms_epsilon",
            ),
            ("ssm.conv_kernel", "ssm.conv_kernel"),
            ("kda.head_dim", "ssm.state_size"),
            ("expert_count", "expert_count"),
            ("expert_shared_count", "expert_shared_count"),
            ("expert_gating_func", "expert_gating_func"),
            ("expert_weights_scale", "expert_weights_scale"),
            ("expert_weights_norm", "expert_weights_norm"),
        ]
        .map(|(key, named)| {
            (
                format!("missing-{key}"),
                with(key, None),
                tensors(),
                format!("solar-open2.{named}"),
            )
        })
        .into_iter()
        .collect();
        assert_all_refused(cases);
    }

    /// A value the reader does not run is refused with the key's name and the value.
    #[test]
    fn unsupported_values_are_refused_by_key_and_value() {
        let few_kv = {
            let mut kv = vec![0i32; LAYERS - 1];
            kv[0] = KV as i32;
            kv
        };
        let cases = [
            (
                "dense-lead",
                with("leading_dense_block_count", Some(V::U32(1))),
                "leading_dense_block_count: is 1",
            ),
            (
                "value-width",
                with("attention.value_length", Some(V::U32(8))),
                "attention.value_length: is 8, attention.key_length 16",
            ),
            (
                "kv-divides",
                with(
                    "attention.head_count_kv",
                    Some(V::I32s(vec![3, 0, 0, 0, 3, 0, 0, 0])),
                ),
                "attention.head_count_kv: is 3 at layer 0 for 4 query heads",
            ),
            (
                "kv-length",
                with("attention.head_count_kv", Some(V::I32s(few_kv))),
                "attention.head_count_kv: has 7 values for 8 layers",
            ),
            (
                "nextn",
                with("nextn_predict_layers", Some(V::U32(1))),
                "nextn_predict_layers: is 1",
            ),
            (
                "softmax-router",
                with("expert_gating_func", Some(V::U32(1))),
                "expert_gating_func: is 1",
            ),
            (
                "expert-groups",
                with("expert_group_count", Some(V::U32(2))),
                "expert_group_count: is 2; the solar-open2 router takes one group",
            ),
            (
                "group-heads",
                with("ssm.group_count", Some(V::U32(8))),
                "ssm.group_count: is 8",
            ),
        ]
        .map(|(tag, kv, want)| (tag.to_string(), kv, tensors(), want.to_string()))
        .into_iter()
        .collect();
        assert_all_refused(cases);
    }

    /// The tensors must be exactly the kinds': a KDA layer with the GQA gate projection, a GQA
    /// layer without it or with a KDA stem, and a layer missing a block tensor are each named, the
    /// missing before the misplaced.
    #[test]
    fn the_tensors_must_match_the_kinds() {
        let mut gate_on_kda = tensors();
        gate_on_kda.push(("blk.1.attn_gate.weight".into(), vec![E, HEADS * K]));
        let mut kda_on_gqa = tensors();
        kda_on_gqa.push(("blk.4.ssm_a".into(), vec![1, HEADS]));
        let cases = [
            (
                "gate-on-kda",
                gate_on_kda,
                "blk.1.attn_gate.weight: is in the file, and its layer's kind carries none",
            ),
            (
                "gate-missing",
                tensors_with("blk.0.attn_gate.weight", None),
                "blk.0.attn_gate.weight: is not in the file",
            ),
            (
                "kda-stem-on-gqa",
                kda_on_gqa,
                "blk.4.ssm_a: is in the file, and its layer's kind carries none",
            ),
            (
                "bias-missing",
                tensors_with("blk.7.exp_probs_b.bias", None),
                "blk.7.exp_probs_b.bias: is not in the file",
            ),
            (
                "head-missing",
                tensors_with("output.weight", None),
                "output.weight: is not in the file",
            ),
        ]
        .map(|(tag, t, want)| (tag.to_string(), keys(), t, want.to_string()))
        .into_iter()
        .collect();
        assert_all_refused(cases);
    }

    /// The tensors hold the dims the keys give. `ssm_a` is stored [1, heads] by llama.cpp's
    /// converter, and ik's [heads, 1] layout is another file than this reader's.
    #[test]
    fn the_dims_are_the_keys() {
        let cases = [
            (
                "ssm-a-ik-layout",
                tensors_with("blk.1.ssm_a", Some(vec![HEADS, 1])),
                "blk.1.ssm_a [4, 1] (want [1, 4])",
            ),
            (
                "gqa-key-width",
                tensors_with("blk.0.attn_k.weight", Some(vec![E, HEADS * K])),
                "blk.0.attn_k.weight [64, 64] (want [64, 32])",
            ),
            (
                "gate-width",
                tensors_with("blk.4.attn_gate.weight", Some(vec![E, K])),
                "blk.4.attn_gate.weight [64, 16] (want [64, 64])",
            ),
            (
                "kda-key-width",
                tensors_with("blk.1.attn_k.weight", Some(vec![E, KV * K])),
                "blk.1.attn_k.weight [64, 32] (want [64, 64])",
            ),
            (
                "conv-taps",
                tensors_with(
                    "blk.2.ssm_conv1d_q.weight",
                    Some(vec![CONV + 1, 1, HEADS * K]),
                ),
                "blk.2.ssm_conv1d_q.weight [5, 1, 64]",
            ),
            (
                "router",
                tensors_with("blk.3.ffn_gate_inp.weight", Some(vec![E, N_EXP + 1])),
                "blk.3.ffn_gate_inp.weight [64, 9] (want [64, 8])",
            ),
            (
                "shared-width",
                tensors_with("blk.5.ffn_up_shexp.weight", Some(vec![E, 2 * EFF])),
                "blk.5.ffn_up_shexp.weight [64, 48] (want [64, 24])",
            ),
        ]
        .map(|(tag, t, want)| (tag.to_string(), keys(), t, want.to_string()))
        .into_iter()
        .collect();
        assert_all_refused(cases);
    }
}
