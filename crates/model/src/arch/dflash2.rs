//! `dflash2` — a DFlash2 draft file: the architecture string `dflash` with a
//! selector head (`selector_hidden.weight`), which tells it apart from V4.1's
//! DSpark draft ([`super::dspark`]) under the same string. Raon-1-mini's draft
//! is one.
//!
//! The draft: `fc` projects the target's rows at `target_layers` (each
//! `target_hidden_size` wide: a hyper-connection target's whole stream set),
//! `enc.output_norm` normalizes the projection, and every layer turns it into
//! its own K and V at the row's position. A block `[id_last, mask × (b − 1)]`
//! then runs through dense Qwen3-shaped layers (GQA with q/k norms, SiLU
//! MLP), each attending to the context and to the whole block (the file's
//! `attention.causal` is false) within the sliding window. Around both
//! sub-layers sits DFlash2's dynamic convolution over the block's rows:
//! `attn_conv_proj`/`ffn_conv_proj` give each row `kernel × 2` coefficients
//! per channel group, added to `*_conv_base`; side 0 mixes the sub-layer's
//! input, side 1 its output. `output_norm` precedes the target's output
//! projection; the selector scores the top `selector_top_k` ids of every
//! block row against the row before (`runtime::selector`). The token
//! embedding and the output projection are the target's and are not in this
//! file.
//!
//! This module reads the file and does not run it: [`Hparams::read`] takes
//! every key it needs, [`names`] owns the tensor names and [`tensors`] states
//! the shape and type of every tensor; [`check`] holds the file to that
//! statement; [`shape`] is the geometry the common DSpark owner runs it by.

use gguf::{GgmlType, Split, Value};
use runtime::dspark::{Context, DraftVocab, DsparkShape, HeadKind};

use super::{DFLASH, meta_arr, meta_bool, meta_f32, meta_usize, metadata, n_vocab, positive};
use crate::placement::PlacementError;

/// The tensor whose presence makes a `dflash` file a DFlash2 draft.
const SELECTOR: &str = "selector_hidden.weight";

/// Keys that would change the block pass or the head's logits (M-RoPE
/// sections, scales, a soft cap); this reader serves none of them and
/// refuses a file that carries one.
const UNSERVED: [&str; 6] = [
    "rope.dimension_sections",
    "embedding_scale",
    "attention.scale",
    "attention.value_scale",
    "logit_scale",
    "final_logit_softcapping",
];

/// Whether `split` is a DFlash2 draft: a `dflash` file with a selector head.
pub fn is_dflash2(split: &Split) -> bool {
    super::is_dflash(split) && split.find(SELECTOR).is_some()
}

/// DFlash2's dynamic convolution around both sub-layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv {
    /// `conv_kernel_size` — the block rows one output row mixes, itself and
    /// the ones before it.
    pub kernel: usize,
    /// `conv_group_size` — channels sharing one coefficient.
    pub group: usize,
}

impl Conv {
    /// Coefficients one row's projection gives: per channel group, `kernel`
    /// taps on each of the two sides.
    #[must_use]
    pub fn coeffs(&self, n_embd: usize) -> usize {
        n_embd / self.group * self.kernel * 2
    }
}

/// DFlash2's selector head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selector {
    /// `selector_rank` — the width of the pairwise score's factors.
    pub rank: usize,
    /// `selector_top_k` — candidates a block row keeps.
    pub top_k: usize,
}

/// The hyperparameters of a DFlash2 draft file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// `block_count`.
    pub n_layer: usize,
    /// `embedding_length` — the draft's width, the target's stream width too.
    pub n_embd: usize,
    /// `feed_forward_length`.
    pub n_ff: usize,
    /// `attention.head_count`.
    pub n_head: usize,
    /// `attention.head_count_kv`.
    pub n_head_kv: usize,
    /// `attention.key_length`, which `attention.value_length` equals.
    pub head_dim: usize,
    /// `rope.freq_base` — NEOX rotation over the whole head.
    pub rope_base: f32,
    /// `attention.layer_norm_rms_epsilon`.
    pub rms_eps: f32,
    /// `attention.sliding_window` — every layer's, as
    /// `attention.sliding_window_pattern` must say.
    pub window: usize,
    /// `context_length`.
    pub n_ctx_train: usize,
    /// `block_size` — rows of one block: the anchor and `block_size − 1`
    /// mask rows.
    pub block_size: usize,
    /// `conv_kernel_size`, `conv_group_size`.
    pub conv: Conv,
    /// `selector_rank`, `selector_top_k`.
    pub selector: Selector,
    /// `target_layers` — the target layers whose entering rows the draft
    /// reads (llama.cpp's layer input: the rows layer `L − 1` leaves).
    pub target_layers: Vec<usize>,
    /// `target_hidden_size` — values of one tapped row.
    pub target_hidden: usize,
    /// `tokenizer.ggml.mask_token_id`.
    pub mask_token: u32,
    /// The target's vocabulary, which the selector's tables cover.
    pub n_vocab: usize,
}

impl Hparams {
    /// Every hyperparameter from `split`'s header. A missing or malformed
    /// key, a file of another architecture, or a value this reader does not
    /// serve (a causal block, a layer outside the window, an activation other
    /// than SiLU, a scale key) is an error naming the key.
    pub fn read(split: &Split) -> Result<Hparams, PlacementError> {
        if !is_dflash2(split) {
            return Err(PlacementError::Metadata {
                key: "architecture".to_string(),
                detail: format!(
                    "is {:?} without {SELECTOR}, not a {DFLASH:?} DFlash2 draft",
                    split.architecture()
                ),
            });
        }
        let n_layer = positive(split, "block_count")?;
        let n_embd = positive(split, "embedding_length")?;
        let head_dim = positive(split, "attention.key_length")?;
        let value_length = meta_usize(split, "attention.value_length")?;
        if value_length != head_dim {
            let detail = format!("is {value_length}, the key length is {head_dim}");
            return Err(metadata(split, "attention.value_length", detail));
        }
        if meta_bool(split, "attention.causal")? {
            let detail = "is true; a DFlash2 block attends to all its rows";
            return Err(metadata(split, "attention.causal", detail));
        }
        windowed(split, n_layer)?;
        activation(split)?;
        for key in UNSERVED {
            if split.value(&split.arch_key(key)).is_some() {
                return Err(metadata(split, key, "is set; this reader serves none"));
            }
        }
        let conv = Conv {
            kernel: positive(split, "conv_kernel_size")?,
            group: positive(split, "conv_group_size")?,
        };
        if !n_embd.is_multiple_of(conv.group) {
            let detail = format!("is {}, which does not divide {n_embd}", conv.group);
            return Err(metadata(split, "conv_group_size", detail));
        }
        let block_size = positive(split, "block_size")?;
        if block_size < 2 {
            return Err(metadata(split, "block_size", "is 1: no row to draft"));
        }
        Ok(Hparams {
            n_layer,
            n_embd,
            n_ff: positive(split, "feed_forward_length")?,
            n_head: positive(split, "attention.head_count")?,
            n_head_kv: positive(split, "attention.head_count_kv")?,
            head_dim,
            rope_base: meta_f32(split, "rope.freq_base")?,
            rms_eps: meta_f32(split, "attention.layer_norm_rms_epsilon")?,
            window: positive(split, "attention.sliding_window")?,
            n_ctx_train: positive(split, "context_length")?,
            block_size,
            conv,
            selector: Selector {
                rank: positive(split, "selector_rank")?,
                top_k: positive(split, "selector_top_k")?,
            },
            target_layers: super::dspark::target_layers(split)?,
            target_hidden: positive(split, "target_hidden_size")?,
            mask_token: super::dspark::mask_token(split)?,
            n_vocab: n_vocab(split)?,
        })
    }

    /// Values of one context row before `fc`: every tap's row, in
    /// `target_layers` order.
    #[must_use]
    pub fn fc_in(&self) -> usize {
        self.target_layers.len() * self.target_hidden
    }
}

/// The draft file's geometry as the common DSpark owner reads it. The taps
/// are the file's `target_layers` as they stand, each tap the whole stream set
/// entering that layer (`target_hidden` values). The block is
/// `[id_last, mask × (b − 1)]` and row 0 is only the anchor, so a block of `b`
/// rows drafts `b − 1` ids; the selector runs over the whole block every time.
/// The selector's tables cover the target's own ids. Pass the result to
/// `DsparkShape::check` before use.
#[must_use]
pub fn shape(hp: &Hparams) -> DsparkShape {
    DsparkShape {
        block: hp.block_size,
        row0_drafts: false,
        taps: hp.target_layers.clone(),
        hidden: hp.target_hidden,
        context: Context::Fc,
        vocab: DraftVocab::Full { n: hp.n_vocab },
        mask_id: hp.mask_token,
        window: Some(hp.window),
        head: HeadKind::Selector {
            rank: hp.selector.rank,
            top_k: hp.selector.top_k,
        },
    }
}

/// Every layer windowed: `attention.sliding_window_pattern` true for each of
/// the first `n_layer` layers.
fn windowed(split: &Split, n_layer: usize) -> Result<(), PlacementError> {
    let key = "attention.sliding_window_pattern";
    let items = meta_arr(split, key)?;
    if items.len() < n_layer {
        let detail = format!("has {} entries for {n_layer} layers", items.len());
        return Err(metadata(split, key, detail));
    }
    if let Some((l, v)) = items[..n_layer]
        .iter()
        .enumerate()
        .find(|(_, v)| v.as_bool() != Some(true))
    {
        let detail = format!("layer {l} is {v:?}; every layer is read windowed");
        return Err(metadata(split, key, detail));
    }
    Ok(())
}

/// `hidden_activation`: absent or `silu` (the reference's default); any
/// other value is refused by name.
fn activation(split: &Split) -> Result<(), PlacementError> {
    let key = "hidden_activation";
    match split.value(&split.arch_key(key)) {
        None => Ok(()),
        Some(Value::String(s)) if s == "silu" => Ok(()),
        Some(v) => Err(metadata(
            split,
            key,
            format!("is {v:?}; only silu is served"),
        )),
    }
}

/// The draft's tensor names.
pub mod names {
    /// `fc` over the taps.
    #[must_use]
    pub fn fc() -> String {
        "fc.weight".to_string()
    }

    /// The norm after `fc`.
    #[must_use]
    pub fn enc_output_norm() -> String {
        "enc.output_norm.weight".to_string()
    }

    /// The norm before the target's output projection.
    #[must_use]
    pub fn output_norm() -> String {
        "output_norm.weight".to_string()
    }

    /// The selector's gate over the head's input row.
    #[must_use]
    pub fn selector_hidden() -> String {
        super::SELECTOR.to_string()
    }

    /// The selector's factor of the row before.
    #[must_use]
    pub fn selector_predecessor() -> String {
        "selector_predecessor.weight".to_string()
    }

    /// The selector's factor of the scored row.
    #[must_use]
    pub fn selector_successor() -> String {
        "selector_successor.weight".to_string()
    }

    /// Layer `l`'s tensor `what` (`attn_q.weight`, `ffn_conv_base`, …).
    #[must_use]
    pub fn blk(l: usize, what: &str) -> String {
        format!("blk.{l}.{what}")
    }
}

/// A tensor the draft reads: its name, shape in ggml's `ne` order, and the
/// type this reader takes it in.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub ty: Ty,
}

/// The types a tensor may have.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    /// A gain or a convolution base: f32.
    F32,
    /// A matrix: Q8_0, the card's draft row kernel's type.
    Q8_0,
}

impl Ty {
    fn takes(self, ty: GgmlType) -> bool {
        match self {
            Ty::F32 => ty == GgmlType::F32,
            Ty::Q8_0 => ty == GgmlType::Q8_0,
        }
    }
}

/// Every tensor of the draft, layer by layer and then the model-level ones,
/// with the shapes `hp` implies.
#[must_use]
pub fn tensors(hp: &Hparams) -> Vec<Tensor> {
    let u = |v: usize| v as u64;
    let embd = u(hp.n_embd);
    let q = u(hp.n_head * hp.head_dim);
    let kv = u(hp.n_head_kv * hp.head_dim);
    let ff = u(hp.n_ff);
    let coeffs = u(hp.conv.coeffs(hp.n_embd));
    let kernel = u(hp.conv.kernel);
    let t = |name: String, dims: Vec<u64>, ty: Ty| Tensor { name, dims, ty };
    let mut out = Vec::new();
    for l in 0..hp.n_layer {
        let b = |what: &str| names::blk(l, what);
        out.extend([
            t(b("attn_norm.weight"), vec![embd], Ty::F32),
            t(b("attn_conv_proj.weight"), vec![embd, coeffs], Ty::Q8_0),
            t(b("attn_conv_base"), vec![embd, kernel, 2], Ty::F32),
            t(b("attn_q.weight"), vec![embd, q], Ty::Q8_0),
            t(b("attn_k.weight"), vec![embd, kv], Ty::Q8_0),
            t(b("attn_v.weight"), vec![embd, kv], Ty::Q8_0),
            t(b("attn_q_norm.weight"), vec![u(hp.head_dim)], Ty::F32),
            t(b("attn_k_norm.weight"), vec![u(hp.head_dim)], Ty::F32),
            t(b("attn_output.weight"), vec![q, embd], Ty::Q8_0),
            t(b("ffn_norm.weight"), vec![embd], Ty::F32),
            t(b("ffn_conv_proj.weight"), vec![embd, coeffs], Ty::Q8_0),
            t(b("ffn_conv_base"), vec![embd, kernel, 2], Ty::F32),
            t(b("ffn_gate.weight"), vec![embd, ff], Ty::Q8_0),
            t(b("ffn_up.weight"), vec![embd, ff], Ty::Q8_0),
            t(b("ffn_down.weight"), vec![ff, embd], Ty::Q8_0),
        ]);
    }
    let rank = u(hp.selector.rank);
    let vocab = u(hp.n_vocab);
    out.extend([
        t(names::fc(), vec![u(hp.fc_in()), embd], Ty::Q8_0),
        t(names::enc_output_norm(), vec![embd], Ty::F32),
        t(names::output_norm(), vec![embd], Ty::F32),
        t(names::selector_hidden(), vec![embd, rank], Ty::Q8_0),
        t(names::selector_predecessor(), vec![rank, vocab], Ty::Q8_0),
        t(names::selector_successor(), vec![rank, vocab], Ty::Q8_0),
    ]);
    out
}

/// Hold `split` to [`tensors`]: every stated tensor present with its shape
/// and type, and no tensor the statement does not name. Each mismatch is an
/// error naming the tensor.
pub fn check(split: &Split, hp: &Hparams) -> Result<(), PlacementError> {
    let stated = tensors(hp);
    for s in &stated {
        let (_, t) = split.find(&s.name).ok_or_else(|| PlacementError::Tensor {
            name: s.name.clone(),
            detail: "is not in the file".to_string(),
        })?;
        if t.dims != s.dims {
            return Err(PlacementError::Tensor {
                name: s.name.clone(),
                detail: format!(
                    "has dims {:?}; the hyperparameters imply {:?}",
                    t.dims, s.dims
                ),
            });
        }
        if !s.ty.takes(t.ty) {
            return Err(PlacementError::Tensor {
                name: s.name.clone(),
                detail: format!("is {:?}; this reader takes {:?}", t.ty, s.ty),
            });
        }
    }
    if let Some((_, t)) = split
        .iter_tensors()
        .find(|(_, t)| !stated.iter().any(|s| s.name == t.name))
    {
        return Err(PlacementError::Tensor {
            name: t.name.clone(),
            detail: "is in the file but not in the DFlash2 statement".to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Conv, Hparams, Selector, Ty, tensors};

    /// Raon-1-mini's draft as its header states it.
    fn raon() -> Hparams {
        Hparams {
            n_layer: 5,
            n_embd: 2560,
            n_ff: 8192,
            n_head: 20,
            n_head_kv: 5,
            head_dim: 128,
            rope_base: 1e7,
            rms_eps: 1e-5,
            window: 2048,
            n_ctx_train: 262_144,
            block_size: 8,
            conv: Conv {
                kernel: 2,
                group: 16,
            },
            selector: Selector {
                rank: 256,
                top_k: 16,
            },
            target_layers: vec![2, 7, 12, 17, 22],
            target_hidden: 10_240,
            mask_token: 0,
            n_vocab: 259_840,
        }
    }

    /// The statement reproduces the file's 81 tensors and their dims as the
    /// header lists them.
    #[test]
    fn statement_matches_the_raon_header() {
        let hp = raon();
        let t = tensors(&hp);
        assert_eq!(t.len(), 81);
        let dims = |name: &str| {
            t.iter()
                .find(|x| x.name == name)
                .map(|x| x.dims.clone())
                .unwrap_or_else(|| panic!("{name} not stated"))
        };
        assert_eq!(dims("fc.weight"), vec![51_200, 2560]);
        assert_eq!(dims("blk.0.attn_conv_proj.weight"), vec![2560, 640]);
        assert_eq!(dims("blk.0.attn_conv_base"), vec![2560, 2, 2]);
        assert_eq!(dims("blk.4.attn_k.weight"), vec![2560, 640]);
        assert_eq!(dims("blk.4.attn_output.weight"), vec![2560, 2560]);
        assert_eq!(dims("selector_successor.weight"), vec![256, 259_840]);
        assert_eq!(dims("selector_hidden.weight"), vec![2560, 256]);
        let f32s = t.iter().filter(|x| x.ty == Ty::F32).count();
        assert_eq!(f32s, 5 * 6 + 2);
    }
}
