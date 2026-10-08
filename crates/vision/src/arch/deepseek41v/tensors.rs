//! The tensors a `deepseek41v` file must hold: every name of [`super::names`] once, at the shape
//! the hyperparameters give it and the type this crate reads, and nothing else.
//!
//! Matrices are bf16 (the checkpoint's own type — the file is a relabelling of it); gains, biases
//! and delimiter rows are f32. A tensor under a name the table does not know is refused by that
//! name before anything else is looked at, so a file of another layout fails on its first
//! unfamiliar tensor, not on a shape. Each row also says where the tensor lives once loaded
//! ([`Home`]): the encoder's card, or the host's span assembly.

use gguf::GgmlType;

use super::{Hparams, names};
use crate::VisionError;
use crate::arch::table;

/// Where a tensor of the file lives once it is loaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Home {
    /// Uploaded with the encoder's weights (`gpu_vision::Encoder::load`).
    Card,
    /// A span delimiter row: the text model's input, read on the host by the span assembly and
    /// never uploaded with the encoder.
    Span,
}

/// One tensor the file must hold, dims in ggml `ne[]` order (the contiguous axis first).
pub type Expected = table::Expected<Home>;

/// Every tensor of a file with these hyperparameters, in the table's order.
#[must_use]
pub fn expected(hp: &Hparams) -> Vec<Expected> {
    let d = |n: usize| n as u64;
    let (dim, ff, p, out) = (d(hp.dim), d(hp.ff), d(hp.patch), d(hp.out_dim));
    let unfold = dim * d(hp.downsample * hp.downsample);
    let mat = |name: String, dims: &[u64]| Expected::row(name, dims, GgmlType::BF16, Home::Card);
    let vec = |name: String, n: u64| Expected::row(name, &[n], GgmlType::F32, Home::Card);
    let span = |name: String| Expected::row(name, &[out], GgmlType::F32, Home::Span);
    let mut all = vec![
        mat(names::patch_embd_weight(), &[p, p, 3, dim]),
        vec(names::patch_embd_bias(), dim),
    ];
    for b in 0..hp.n_layer {
        all.extend([
            vec(names::ln1(b), dim),
            mat(names::attn_qkv_weight(b), &[dim, 3 * dim]),
            vec(names::attn_qkv_bias(b), 3 * dim),
            mat(names::attn_out_weight(b), &[dim, dim]),
            vec(names::attn_out_bias(b), dim),
            vec(names::ln2(b), dim),
            mat(names::ffn_gate(b), &[dim, ff]),
            mat(names::ffn_up(b), &[dim, ff]),
            mat(names::ffn_down(b), &[ff, dim]),
        ]);
    }
    all.extend([
        vec(names::post_ln(), dim),
        mat(names::mm1_weight(), &[unfold, out]),
        vec(names::mm1_bias(), out),
        mat(names::mm2_weight(), &[out, out]),
        vec(names::mm2_bias(), out),
        span(names::img_start()),
        span(names::img_end()),
        span(names::image_newline()),
    ]);
    all
}

/// Check a file's tensors (name, ggml dims, type) against [`expected`]: every one known, none
/// twice, each at its shape and type, and none of the table missing. Returns how many were named.
pub fn check<'a>(
    hp: &Hparams,
    tensors: impl IntoIterator<Item = (&'a str, &'a [u64], GgmlType)>,
) -> Result<usize, VisionError> {
    table::check(super::PROJECTOR_TYPE, &expected(hp), tensors)
}

#[cfg(test)]
pub(super) mod tests {
    use super::{check, expected};
    use crate::arch::deepseek41v::Hparams;
    use crate::arch::table::testing::{changes_are_refused_by_name, text, triples};

    /// The V4.1 file's hyperparameters.
    pub(in crate::arch::deepseek41v) fn hp() -> Hparams {
        Hparams {
            n_layer: 32,
            dim: 1024,
            n_head: 16,
            ff: 2816,
            patch: 14,
            downsample: 3,
            max_tokens: 1024,
            min_pixels: 295_936,
            out_dim: 5120,
            eps: 1e-6,
            rope_theta: 10_000.0,
        }
    }

    /// 9 tensors per block and 10 others: the 298 of the V4.1 file.
    #[test]
    fn table_is_298() {
        assert_eq!(expected(&hp()).len(), 32 * 9 + 10);
    }

    /// The table itself passes; one renamed tensor is refused by its new name, a dropped one is
    /// named as missing, a doubled one as twice, a reshaped one by its dims.
    #[test]
    fn a_changed_tensor_list_is_refused_by_name() {
        let hp = hp();
        changes_are_refused_by_name(
            super::super::PROJECTOR_TYPE,
            &expected(&hp),
            298,
            "v.image_newline",
            |l| text(check(&hp, triples(l))),
        );
    }
}
