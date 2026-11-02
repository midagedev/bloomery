//! The tensors a `pixtral` file must hold: every name of [`super::names`] once, at the shape the
//! hyperparameters give it and the type this crate reads, and nothing else.
//!
//! The file is a bf16 export of llama.cpp's converter: the block matrices and the projector's two
//! linears are bf16; every norm gain, the patch kernel, the patch merger and the `[IMG_BREAK]` row
//! are f32 (the converter keeps them out of the narrow type). A file of another export is refused
//! by the first tensor whose type differs.

use gguf::GgmlType;

use super::{Hparams, names};
use crate::VisionError;
use crate::arch::table;

/// One tensor the file must hold, dims in ggml `ne[]` order (the contiguous axis first). The
/// file's tensors all load the same way, so a row carries no note on where it lives.
pub type Expected = table::Expected<()>;

/// Every tensor of a file with these hyperparameters, in the table's order.
#[must_use]
pub fn expected(hp: &Hparams) -> Vec<Expected> {
    let d = |n: usize| n as u64;
    let (dim, ff, p, out) = (d(hp.dim), d(hp.ff), d(hp.patch), d(hp.out_dim));
    let merged = dim * d(hp.merge * hp.merge);
    let mat = |name: String, dims: &[u64]| Expected::row(name, dims, GgmlType::BF16, ());
    let f32s = |name: String, dims: &[u64]| Expected::row(name, dims, GgmlType::F32, ());
    let mut all = vec![
        f32s(names::patch_merger_weight(), &[merged, dim]),
        f32s(names::input_norm_weight(), &[dim]),
        f32s(names::img_break(), &[out]),
        f32s(names::pre_ln_weight(), &[dim]),
        f32s(names::patch_embd_weight(), &[p, p, 3, dim]),
    ];
    for b in 0..hp.n_layer {
        all.extend([
            f32s(names::ln1_weight(b), &[dim]),
            mat(names::attn_q_weight(b), &[dim, dim]),
            mat(names::attn_k_weight(b), &[dim, dim]),
            mat(names::attn_v_weight(b), &[dim, dim]),
            mat(names::attn_out_weight(b), &[dim, dim]),
            f32s(names::ln2_weight(b), &[dim]),
            mat(names::ffn_gate_weight(b), &[dim, ff]),
            mat(names::ffn_up_weight(b), &[dim, ff]),
            mat(names::ffn_down_weight(b), &[ff, dim]),
        ]);
    }
    all.extend([
        mat(names::mm1_weight(), &[dim, out]),
        mat(names::mm2_weight(), &[out, out]),
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
pub(crate) mod tests {
    use gguf::GgmlType;

    use super::{check, expected};
    use crate::arch::pixtral::{Hparams, PROJECTOR_TYPE, names};
    use crate::arch::table::testing::{changes_are_refused_by_name, list, text, triples};
    use crate::preprocess::CLIP;

    /// Mistral Large 4's file.
    pub(crate) fn hp() -> Hparams {
        Hparams {
            n_layer: 30,
            dim: 2048,
            n_head: 16,
            ff: 6144,
            patch: 14,
            merge: 2,
            out_dim: 6144,
            eps: 1e-5,
            stats: CLIP,
        }
    }

    /// 5 tensors outside the blocks, 9 a block and 2 of the projector: the 277 tensors of the file,
    /// whose table of contents sums to the 3,444,678,656 B of data the file holds (its tensors'
    /// `n_bytes` as gguf-py reads them).
    #[test]
    fn the_table_is_the_files() {
        let all = expected(&hp());
        assert_eq!(all.len(), 5 + 30 * 9 + 2);
        assert_eq!(all.len(), 277);
        let bytes: u64 = all.iter().map(|e| e.bytes()).sum();
        assert_eq!(bytes, 3_444_678_656);
    }

    #[test]
    fn the_file_checks_against_the_table() {
        let all = expected(&hp());
        let l = list(&all);
        assert_eq!(check(&hp(), triples(&l)).expect("table checks"), 277);
    }

    /// Every kind of change is refused by the tensor's name.
    #[test]
    fn changes_are_refused_by_name_of_the_tensor() {
        let all = expected(&hp());
        changes_are_refused_by_name(PROJECTOR_TYPE, &all, 277, &names::mm2_weight(), |l| {
            text(check(&hp(), triples(l)))
        });
    }

    /// The file's matrices are bf16: an f16 export is refused by the first matrix's name and type.
    #[test]
    fn an_f16_export_is_refused_by_name() {
        let all = expected(&hp());
        let mut l = list(&all);
        let q = names::attn_q_weight(0);
        let at = l.iter().position(|r| r.0 == q).expect("row");
        l[at].2 = GgmlType::F16;
        assert_eq!(
            text(check(&hp(), triples(&l))).unwrap_err(),
            format!("tensor {q}: is f16; this crate reads it as bf16")
        );
    }
}
