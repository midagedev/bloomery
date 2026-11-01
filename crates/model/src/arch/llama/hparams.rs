//! Every hyperparameter the dense Llama decode step reads, resolved once at
//! load from the file's metadata, and a refusal for every value this chain
//! does not run. This is the one reader of the `llama.*` keys and of the
//! `qwen3.*` keys, which spell the same facts under their own architecture's
//! name; the one difference between the two families, how the rope pairs a
//! head's values, is the reader's `pairing` argument.
//!
//! A key ik reads as optional keeps ik's reading of its absence, and the
//! reading is recorded in [`Hparams::defaults`] with the place ik sets it: the
//! description never carries a value the file did not give without saying so.
//! The places are ik's `src/llama-hparams.cpp` (the tree
//! `tools/ref/models/qwen3moe.sh` names as `IK`).

use gguf::Split;

use crate::arch::classify::rope_factors;
use crate::arch::{finite_positive, metadata, n_vocab, optional_usize, plain_rope, positive};
use crate::placement::PlacementError;

/// ik's `rope_freq_base_train` before `rope.freq_base` is read
/// (llama-hparams.cpp).
const IK_ROPE_BASE: f32 = 10_000.0;

/// The hyperparameters of one llama file this chain runs.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// `block_count`.
    pub n_layer: usize,
    /// `embedding_length` — the residual width.
    pub n_embd: usize,
    /// `feed_forward_length` — the SwiGLU block's hidden width, every layer's.
    pub n_ff: usize,
    /// `attention.head_count` — query heads.
    pub n_head: usize,
    /// `attention.head_count_kv` — key and value heads; `n_head / n_head_kv`
    /// query heads share each one.
    pub n_head_kv: usize,
    /// `attention.key_length`, which `attention.value_length` equals: one
    /// head's query, key and value width.
    pub head_dim: usize,
    /// The rope every layer applies to Q and K.
    pub rope: Rope,
    /// `attention.layer_norm_rms_epsilon` — every RMS norm's.
    pub rms_eps: f32,
    /// The vocabulary size: `vocab_size` when the file carries it, else the
    /// length of `tokenizer.ggml.tokens` (ik's rule, llama-hparams.cpp).
    pub n_vocab: usize,
    /// `context_length` — the context the model was trained to, not the one
    /// this engine allocates.
    pub n_ctx_train: usize,
    /// The keys the file does not carry that ik reads as optional, each with
    /// the value taken and the place that sets it.
    pub defaults: Vec<String>,
}

/// How the rope pairs a head's values: the one fact the two families of this
/// reader differ by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeMode {
    /// Value `2i` rotates with value `2i + 1` — `LLAMA_ROPE_TYPE_NORM`, ik's
    /// rope type for this architecture (`llama_rope_type`, src/llama.cpp: the
    /// group of architectures that "use what we call a normal RoPE").
    Norm,
    /// Value `i` rotates with value `i + dims / 2` — `LLAMA_ROPE_TYPE_NEOX`,
    /// ik's rope type for `qwen3` (`llama_rope_type`, src/llama.cpp).
    Neox,
}

/// The rope ik's graph applies to every layer's Q and K: `ggml_rope_ext` with
/// the file's frequency factors when it carries them, no YaRN (the file carries
/// no scaling keys this chain accepts) and a frequency scale of 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rope {
    pub mode: RopeMode,
    /// `rope.freq_base` — θ.
    pub base: f32,
    /// The rotated width: `rope.dimension_count`; ik's builder asserts it is
    /// the whole head (`build_llama`'s `n_embd_head == n_rot`).
    pub dims: usize,
    /// Whether the file carries `rope_freqs.weight`: one factor for each pair of
    /// the rotated values, which divides the pair's frequency.
    pub factors: bool,
}

impl Hparams {
    /// Read the file's hyperparameters, refusing every value this chain does
    /// not run, its rope pairing `pairing` — the family whose keys the file
    /// carries. Load-time only.
    pub fn read(split: &Split, pairing: RopeMode) -> Result<Hparams, PlacementError> {
        let mut defaults = Vec::new();
        let n_layer = positive(split, "block_count")?;
        let n_embd = positive(split, "embedding_length")?;
        let n_ff = positive(split, "feed_forward_length")?;
        let n_head = positive(split, "attention.head_count")?;
        let n_head_kv = match optional_usize(split, "attention.head_count_kv")? {
            Some(n) => n,
            None => {
                defaults.push(format!(
                    "attention.head_count_kv = {n_head} (llama-hparams.cpp: n_head_kv takes n_head)"
                ));
                n_head
            }
        };
        if n_head_kv == 0 || !n_head.is_multiple_of(n_head_kv) {
            return Err(metadata(
                split,
                "attention.head_count_kv",
                format!(
                    "is {n_head_kv} for {n_head} query heads; grouped-query attention needs a \
                     whole number of query heads per key head"
                ),
            ));
        }
        let key = head_width(split, "attention.key_length", n_embd, n_head, &mut defaults)?;
        let value = head_width(
            split,
            "attention.value_length",
            n_embd,
            n_head,
            &mut defaults,
        )?;
        if key != value || key == 0 {
            return Err(metadata(
                split,
                "attention.value_length",
                format!("is {value}, attention.key_length {key}; one width serves Q, K and V"),
            ));
        }
        // ik reads the count as optional and takes 0 (a dense file); a routed
        // llama file (a mixture of experts) has a router this chain does not run.
        if let Some(n) = optional_usize(split, "expert_count")?
            && n != 0
        {
            return Err(metadata(
                split,
                "expert_count",
                format!("is {n}; this chain runs the dense trunk, no routed experts"),
            ));
        }
        Ok(Hparams {
            n_layer,
            n_embd,
            n_ff,
            n_head,
            n_head_kv,
            head_dim: key,
            rope: Rope::read(split, key, pairing, &mut defaults)?,
            rms_eps: finite_positive(split, "attention.layer_norm_rms_epsilon")?,
            n_vocab: n_vocab(split)?,
            n_ctx_train: positive(split, "context_length")?,
            defaults,
        })
    }

    /// Query heads per key and value head.
    #[must_use]
    pub fn group(&self) -> usize {
        self.n_head / self.n_head_kv
    }
}

/// `suffix`, a head width: the key's value, else ik's reading of its absence
/// (`n_embd / n_head`, llama-hparams.cpp), recorded in `defaults`.
fn head_width(
    split: &Split,
    suffix: &str,
    n_embd: usize,
    n_head: usize,
    defaults: &mut Vec<String>,
) -> Result<usize, PlacementError> {
    if let Some(width) = optional_usize(split, suffix)? {
        return Ok(width);
    }
    if !n_embd.is_multiple_of(n_head) {
        return Err(metadata(
            split,
            suffix,
            format!(
                "is absent, and embedding_length {n_embd} is not a multiple of \
                 attention.head_count {n_head}"
            ),
        ));
    }
    let width = n_embd / n_head;
    defaults.push(format!(
        "{suffix} = {width} (llama-hparams.cpp: n_embd / n_head)"
    ));
    Ok(width)
}

impl Rope {
    fn read(
        split: &Split,
        head_dim: usize,
        pairing: RopeMode,
        defaults: &mut Vec<String>,
    ) -> Result<Rope, PlacementError> {
        let dims = match optional_usize(split, "rope.dimension_count")? {
            Some(dims) => dims,
            None => {
                defaults.push(format!(
                    "rope.dimension_count = {head_dim} (llama-hparams.cpp: n_rot takes the key width)"
                ));
                head_dim
            }
        };
        if dims != head_dim {
            return Err(metadata(
                split,
                "rope.dimension_count",
                format!("is {dims}, the head {head_dim}; this chain rotates whole heads"),
            ));
        }
        plain_rope(split)?;
        let base = match split.value(&split.arch_key("rope.freq_base")) {
            Some(_) => finite_positive(split, "rope.freq_base")?,
            None => {
                defaults.push(format!(
                    "rope.freq_base = {IK_ROPE_BASE} (llama-hparams.cpp: the train base before the key)"
                ));
                IK_ROPE_BASE
            }
        };
        Ok(Rope {
            mode: pairing,
            base,
            dims,
            factors: rope_factors(split, dims)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Hparams, Rope, RopeMode};
    use crate::arch::synthetic::{V, header};
    use gguf::Split;

    /// The keys this module reads, with the values the Nanbeige4.3-33B file's
    /// header holds.
    fn nanbeige() -> Vec<(&'static str, V)> {
        vec![
            ("block_count", V::U32(52)),
            ("context_length", V::U32(524_288)),
            ("embedding_length", V::U32(7168)),
            ("feed_forward_length", V::U32(22_016)),
            ("attention.head_count", V::U32(56)),
            ("attention.head_count_kv", V::U32(8)),
            ("rope.freq_base", V::F32(120_000_000.0)),
            ("attention.layer_norm_rms_epsilon", V::F32(1e-5)),
            ("attention.key_length", V::U32(128)),
            ("attention.value_length", V::U32(128)),
            ("vocab_size", V::U32(166_912)),
            ("rope.dimension_count", V::U32(128)),
        ]
    }

    /// `nanbeige()` with `key` set to `v` (added when absent).
    fn with(key: &'static str, v: V) -> Vec<(&'static str, V)> {
        let mut t = nanbeige();
        match t.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 = v,
            None => t.push((key, v)),
        }
        t
    }

    /// `nanbeige()` without the keys `gone`.
    fn without(gone: &[&str]) -> Vec<(&'static str, V)> {
        let mut t = nanbeige();
        t.retain(|(k, _)| !gone.contains(k));
        t
    }

    /// A header-only llama file of `kv` (a two-token vocabulary), read.
    fn read(tag: &str, kv: &[(&str, V)]) -> Result<Hparams, String> {
        read_as(tag, "llama", RopeMode::Norm, kv)
    }

    /// [`read`] at architecture `arch` and rope pairing `pairing`.
    fn read_as(
        tag: &str,
        arch: &'static str,
        pairing: RopeMode,
        kv: &[(&str, V)],
    ) -> Result<Hparams, String> {
        let path = header(&format!("llama-hparams-{tag}"), arch, kv, &[]);
        let split = Split::open(&path).map_err(|e| e.to_string())?;
        let hp = Hparams::read(&split, pairing).map_err(|e| e.to_string());
        let _ = std::fs::remove_file(&path);
        hp
    }

    /// The Nanbeige header reads, to the values its chain runs, with no key
    /// taken from ik's reading of an absence.
    #[test]
    fn nanbeige_reads() {
        let hp = read("nanbeige", &nanbeige()).expect("the Nanbeige header reads");
        assert_eq!(
            hp,
            Hparams {
                n_layer: 52,
                n_embd: 7168,
                n_ff: 22_016,
                n_head: 56,
                n_head_kv: 8,
                head_dim: 128,
                rope: Rope {
                    mode: RopeMode::Norm,
                    base: 120_000_000.0,
                    dims: 128,
                    factors: false,
                },
                rms_eps: 1e-5,
                n_vocab: 166_912,
                n_ctx_train: 524_288,
                defaults: Vec::new(),
            }
        );
        assert_eq!(hp.group(), 7);
        // The values the file may also spell out, spelled as this chain runs.
        for (i, (k, v)) in [
            ("expert_count", V::U32(0)),
            ("rope.scaling.type", V::Str("none")),
            ("rope.scaling.factor", V::F32(1.0)),
        ]
        .into_iter()
        .enumerate()
        {
            let got = read(&format!("spelled{i}"), &with(k, v)).expect(k);
            assert_eq!(got, hp, "{k} spelled out as this chain runs it");
        }
    }

    /// A qwen3 file spells the same facts under its own architecture's name and
    /// reads with the NEOX pairing, everything else the llama reading's: the
    /// 8B's counts, and no key taken from an absence. Mutant: the pairing
    /// argument dropped (the mode is the family's, not the file's).
    #[test]
    fn a_qwen3_file_reads_the_same_keys_with_the_neox_pairing() {
        let qwen3_8b = vec![
            ("block_count", V::U32(36)),
            ("context_length", V::U32(40_960)),
            ("embedding_length", V::U32(4096)),
            ("feed_forward_length", V::U32(12_288)),
            ("attention.head_count", V::U32(32)),
            ("attention.head_count_kv", V::U32(8)),
            ("rope.freq_base", V::F32(1_000_000.0)),
            ("attention.layer_norm_rms_epsilon", V::F32(1e-6)),
            ("attention.key_length", V::U32(128)),
            ("attention.value_length", V::U32(128)),
            ("vocab_size", V::U32(151_936)),
            ("rope.dimension_count", V::U32(128)),
        ];
        let hp =
            read_as("qwen3-8b", "qwen3", RopeMode::Neox, &qwen3_8b).expect("the 8B's keys read");
        assert_eq!(hp.rope.mode, RopeMode::Neox);
        assert_eq!(
            (
                hp.n_layer,
                hp.n_embd,
                hp.n_ff,
                hp.n_head,
                hp.n_head_kv,
                hp.head_dim
            ),
            (36, 4096, 12_288, 32, 8, 128)
        );
        assert_eq!(
            (hp.rope.base, hp.rope.dims, hp.rope.factors),
            (1_000_000.0, 128, false)
        );
        assert_eq!(hp.group(), 4);
        assert!(hp.defaults.is_empty(), "{:?}", hp.defaults);
    }

    /// A key ik reads as optional and the file leaves out takes ik's reading,
    /// and the description says so.
    #[test]
    fn an_absent_optional_key_is_recorded_with_ik_s_reading() {
        let kv = without(&[
            "attention.head_count_kv",
            "attention.key_length",
            "attention.value_length",
            "rope.dimension_count",
            "rope.freq_base",
        ]);
        let hp = read("absent", &kv).expect("the optional keys may be absent");
        assert_eq!((hp.n_head_kv, hp.head_dim, hp.rope.dims), (56, 128, 128));
        assert_eq!(hp.rope.base, 10_000.0);
        assert_eq!(
            hp.defaults,
            [
                "attention.head_count_kv = 56 (llama-hparams.cpp: n_head_kv takes n_head)",
                "attention.key_length = 128 (llama-hparams.cpp: n_embd / n_head)",
                "attention.value_length = 128 (llama-hparams.cpp: n_embd / n_head)",
                "rope.dimension_count = 128 (llama-hparams.cpp: n_rot takes the key width)",
                "rope.freq_base = 10000 (llama-hparams.cpp: the train base before the key)",
            ]
        );
    }

    /// Each value this chain does not run is refused, and the error names the
    /// key and the value the file holds.
    #[test]
    fn unsupported_values_are_refused_by_key_and_value() {
        let mut odd_width = without(&["attention.key_length", "attention.value_length"]);
        for (k, v) in odd_width.iter_mut() {
            if *k == "embedding_length" {
                *v = V::U32(7169);
            }
        }
        let cases = [
            (
                with("attention.head_count_kv", V::U32(5)),
                "metadata llama.attention.head_count_kv: is 5 for 56 query heads;",
            ),
            (
                with("attention.value_length", V::U32(64)),
                "metadata llama.attention.value_length: is 64, attention.key_length 128;",
            ),
            (
                odd_width,
                "metadata llama.attention.key_length: is absent, and embedding_length 7169 is \
                 not a multiple of attention.head_count 56",
            ),
            (
                with("rope.dimension_count", V::U32(64)),
                "metadata llama.rope.dimension_count: is 64, the head 128;",
            ),
            (
                with("rope.scaling.type", V::Str("llama3")),
                "metadata llama.rope.scaling.type: is \"llama3\";",
            ),
            (
                with("rope.scaling.factor", V::F32(8.0)),
                "metadata llama.rope.scaling.factor: is 8;",
            ),
            (
                with("expert_count", V::U32(8)),
                "metadata llama.expert_count: is 8;",
            ),
            (
                with("feed_forward_length", V::U32(0)),
                "metadata llama.feed_forward_length: is 0",
            ),
            (
                without(&["attention.layer_norm_rms_epsilon"]),
                "metadata llama.attention.layer_norm_rms_epsilon: is absent or not a float",
            ),
            (
                with("rope.freq_base", V::F32(0.0)),
                "metadata llama.rope.freq_base: is 0, not a finite positive value",
            ),
        ];
        let wrong: Vec<String> = cases
            .into_iter()
            .enumerate()
            .filter_map(|(i, (kv, want))| match read(&format!("refuse{i}"), &kv) {
                Ok(_) => Some(format!("case {i} read; expected \"{want}\"")),
                Err(e) if e.starts_with(want) => None,
                Err(e) => Some(format!("case {i}: {e}")),
            })
            .collect();
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }
}
