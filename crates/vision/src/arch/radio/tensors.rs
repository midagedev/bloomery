//! The tensors a `nemotron_omni` file must hold: every name of [`super::names`] once, at the shape
//! the hyperparameters give it and the type this crate reads, and nothing else.
//!
//! The file is a bf16 export of llama.cpp's converter: the block matrices and the projector's two
//! matrices are bf16; every vector, the prefix rows, the position table, the projector's norm gain
//! and the two patch kernels are f32 (the converter keeps the patch kernels out of the narrow
//! type). A file of another export is refused by the first tensor whose type differs.

use gguf::GgmlType;

use super::{Hparams, POS_SIDE, PREFIX_ROWS, PROJECTOR_HIDDEN, names};
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
    let vec = |name: String, n: u64| Expected::row(name, &[n], GgmlType::F32, ());
    let f32s = |name: String, dims: &[u64]| Expected::row(name, dims, GgmlType::F32, ());
    let mut all = vec![
        f32s(names::patch_embd_weight(), &[p, p, 3, dim]),
        f32s(names::patch_embd_weight_video(), &[p, p, 6, dim]),
        f32s(names::class_embd(), &[dim, d(PREFIX_ROWS)]),
        f32s(names::position_embd(), &[dim, d(POS_SIDE * POS_SIDE)]),
    ];
    for b in 0..hp.n_layer {
        all.extend([
            vec(names::ln1_weight(b), dim),
            vec(names::ln1_bias(b), dim),
            mat(names::attn_q_weight(b), &[dim, dim]),
            vec(names::attn_q_bias(b), dim),
            mat(names::attn_k_weight(b), &[dim, dim]),
            vec(names::attn_k_bias(b), dim),
            mat(names::attn_v_weight(b), &[dim, dim]),
            vec(names::attn_v_bias(b), dim),
            mat(names::attn_out_weight(b), &[dim, dim]),
            vec(names::attn_out_bias(b), dim),
            vec(names::ln2_weight(b), dim),
            vec(names::ln2_bias(b), dim),
            mat(names::ffn_up_weight(b), &[dim, ff]),
            vec(names::ffn_up_bias(b), ff),
            mat(names::ffn_down_weight(b), &[ff, dim]),
            vec(names::ffn_down_bias(b), dim),
        ]);
    }
    all.extend([
        vec(names::mlp_norm(), merged),
        mat(names::mlp_up(), &[merged, d(PROJECTOR_HIDDEN)]),
        mat(names::mlp_down(), &[d(PROJECTOR_HIDDEN), out]),
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
mod tests {
    use gguf::GgmlType;

    use super::{check, expected};
    use crate::arch::radio::{Hparams, PROJECTOR_TYPE, names};
    use crate::arch::table::testing::{changes_are_refused_by_name, list, text, triples};
    use crate::preprocess::ChannelStats;

    /// Bolt's file.
    fn hp() -> Hparams {
        Hparams {
            n_layer: 32,
            dim: 1280,
            n_head: 16,
            ff: 5120,
            patch: 16,
            merge: 2,
            out_dim: 4096,
            eps: 1e-6,
            stats: ChannelStats::HALF,
        }
    }

    /// 4 tensors outside the blocks, 16 a block and 3 of the projector: the 519 tensors of the
    /// file, whose table of contents sums to the bytes of the data the file holds.
    #[test]
    fn the_table_is_bolts_file() {
        let all = expected(&hp());
        assert_eq!(all.len(), 4 + 32 * 16 + 3);
        assert_eq!(all.len(), 519);
        let bytes: u64 = all.iter().map(|e| e.bytes()).sum();
        assert_eq!(bytes, 1_733_662_720);
    }

    #[test]
    fn the_file_checks_against_the_table() {
        let all = expected(&hp());
        let l = list(&all);
        assert_eq!(check(&hp(), triples(&l)).expect("table checks"), 519);
    }

    /// Every kind of change is refused by the tensor's name.
    #[test]
    fn changes_are_refused_by_name_of_the_tensor() {
        let all = expected(&hp());
        changes_are_refused_by_name(PROJECTOR_TYPE, &all, 519, &names::mlp_down(), |l| {
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
