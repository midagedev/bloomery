//! The tensors a `minicpmv4_7` file must hold: every name of [`super::names`] once, at the shape
//! the hyperparameters give it and the type this crate reads, and nothing else (460 for the
//! 35B-A3B file).
//!
//! The file is a bf16 export of llama.cpp's converter: matrices are bf16; biases, norm gains, the
//! patch kernel, the position table and the separator rows are f32. Dims are ggml `ne[]` order
//! (the contiguous axis first).

use gguf::GgmlType;

use super::{Hparams, names};
use crate::VisionError;
use crate::arch::table;

/// Rows of the learned position table: a 70×70 grid, the 980-pixel image of the checkpoint.
const POSITIONS: u64 = 4900;

/// Rows of `v.tok_embd_sep`.
const SEPARATORS: u64 = 4;

/// One tensor the file must hold. The file's tensors all load the same way, so a row carries no
/// note on where it lives.
pub type Expected = table::Expected<()>;

/// How the converter typed the file's matrices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Export {
    Bf16,
}

impl Export {
    /// The export a file is, from the type of its `mm.up.weight`.
    #[must_use]
    pub fn of(projector_weight: GgmlType) -> Option<Export> {
        (projector_weight == GgmlType::BF16).then_some(Export::Bf16)
    }

    fn matrix(self) -> GgmlType {
        match self {
            Export::Bf16 => GgmlType::BF16,
        }
    }
}

/// Every tensor of an `export` file with these hyperparameters, in the table's order.
#[must_use]
pub fn expected(hp: &Hparams, export: Export) -> Vec<Expected> {
    let d = |n: usize| n as u64;
    let (dim, ff, p, out) = (d(hp.dim), d(hp.ff), d(hp.patch), d(hp.out_dim));
    let merged = d(hp.merged_dim());
    let mat = |name: String, dims: &[u64]| Expected::row(name, dims, export.matrix(), ());
    let vec = |name: String, n: u64| Expected::row(name, &[n], GgmlType::F32, ());
    let norm = |all: &mut Vec<Expected>, w: String, b: String, n: u64| {
        all.extend([vec(w, n), vec(b, n)]);
    };
    let mut all = vec![
        Expected::row(names::tok_embd_sep(), &[out, SEPARATORS], GgmlType::F32, ()),
        vec(names::patch_embd_bias(), dim),
        Expected::row(
            names::patch_embd_weight(),
            &[p, p, 3, dim],
            GgmlType::F32,
            (),
        ),
        Expected::row(names::position_embd(), &[dim, POSITIONS], GgmlType::F32, ()),
    ];
    for b in 0..hp.n_layer {
        norm(&mut all, names::ln1_weight(b), names::ln1_bias(b), dim);
        norm(&mut all, names::ln2_weight(b), names::ln2_bias(b), dim);
        all.extend([
            vec(names::ffn_up_bias(b), ff),
            mat(names::ffn_up_weight(b), &[dim, ff]),
            vec(names::ffn_down_bias(b), dim),
            mat(names::ffn_down_weight(b), &[ff, dim]),
        ]);
        all.extend([
            vec(names::attn_k_bias(b), dim),
            mat(names::attn_k_weight(b), &[dim, dim]),
            vec(names::attn_out_bias(b), dim),
            mat(names::attn_out_weight(b), &[dim, dim]),
            vec(names::attn_q_bias(b), dim),
            mat(names::attn_q_weight(b), &[dim, dim]),
            vec(names::attn_v_bias(b), dim),
            mat(names::attn_v_weight(b), &[dim, dim]),
        ]);
    }
    norm(
        &mut all,
        names::post_ln_weight(),
        names::post_ln_bias(),
        dim,
    );
    let mm = |stem| (names::mm_weight(stem), names::mm_bias(stem));
    all.extend([
        vec(names::mm_bias("up"), merged),
        mat(names::mm_weight("up"), &[merged, merged]),
        vec(names::mm_bias("down"), out),
        mat(names::mm_weight("down"), &[merged, out]),
    ]);
    let (w, b) = mm("input_norm");
    norm(&mut all, w, b, merged);
    let vm = |stem| (names::vit_merger_weight(stem), names::vit_merger_bias(stem));
    let (w, b) = vm("ln1");
    norm(&mut all, w, b, dim);
    let ds_ff = d(4 * hp.ff);
    all.extend([
        vec(names::vit_merger_bias("ds_ffn_up"), ds_ff),
        mat(names::vit_merger_weight("ds_ffn_up"), &[merged, ds_ff]),
        vec(names::vit_merger_bias("ds_ffn_down"), dim),
        mat(names::vit_merger_weight("ds_ffn_down"), &[ds_ff, dim]),
    ]);
    let (w, b) = vm("ds_ln");
    norm(&mut all, w, b, merged);
    for stem in ["attn_k", "attn_out", "attn_q", "attn_v"] {
        all.extend([
            vec(names::vit_merger_bias(stem), dim),
            mat(names::vit_merger_weight(stem), &[dim, dim]),
        ]);
    }
    all
}

/// Check a file's tensors (name, ggml dims, type) against [`expected`] for the export the file
/// is: every one known, none twice, each at its shape and type, and none of the table missing.
/// Returns how many were named.
pub fn check<'a>(
    hp: &Hparams,
    tensors: impl IntoIterator<Item = (&'a str, &'a [u64], GgmlType)>,
) -> Result<usize, VisionError> {
    let tensors: Vec<(&str, &[u64], GgmlType)> = tensors.into_iter().collect();
    let up = names::mm_weight("up");
    let err = |detail: String| VisionError::Tensor {
        name: up.clone(),
        detail,
    };
    let Some(&(_, _, ty)) = tensors.iter().find(|(name, ..)| *name == up) else {
        return Err(err("is missing from the file".into()));
    };
    let Some(export) = Export::of(ty) else {
        return Err(err(format!(
            "is {ty}; the matrices of a {} file are bf16",
            super::PROJECTOR_TYPE
        )));
    };
    table::check(super::PROJECTOR_TYPE, &expected(hp, export), tensors)
}

#[cfg(test)]
mod tests {
    use gguf::GgmlType;

    use super::{Export, check, expected};
    use crate::arch::minicpmv::hparams::tests::hp;
    use crate::arch::table::testing::{changes_are_refused_by_name, list, row_of, text, triples};

    /// The 35B-A3B file's 460 tensors: 28 outside the blocks (4 before them, 2 after, 6 of the
    /// final merger, 16 of the ViT merger) and 16 a block.
    #[test]
    fn the_table_has_the_files_460_tensors() {
        let t = expected(&hp(), Export::Bf16);
        assert_eq!(t.len(), 460);
        assert_eq!(t.len(), 28 + 27 * 16);
        let l = list(&t);
        // Spot rows of the header dump: shape and type.
        let at = |name: &str| {
            let (_, dims, ty) = &l[row_of(&l, name)];
            (dims.clone(), *ty)
        };
        assert_eq!(at("v.tok_embd_sep"), (vec![2048, 4], GgmlType::F32));
        assert_eq!(
            at("v.patch_embd.weight"),
            (vec![14, 14, 3, 1152], GgmlType::F32)
        );
        assert_eq!(
            at("v.position_embd.weight"),
            (vec![1152, 4900], GgmlType::F32)
        );
        assert_eq!(
            at("v.blk.0.ffn_up.weight"),
            (vec![1152, 4304], GgmlType::BF16)
        );
        assert_eq!(
            at("v.blk.26.ffn_down.weight"),
            (vec![4304, 1152], GgmlType::BF16)
        );
        assert_eq!(at("v.blk.5.attn_q.bias"), (vec![1152], GgmlType::F32));
        assert_eq!(at("mm.up.weight"), (vec![4608, 4608], GgmlType::BF16));
        assert_eq!(at("mm.down.weight"), (vec![4608, 2048], GgmlType::BF16));
        assert_eq!(at("mm.down.bias"), (vec![2048], GgmlType::F32));
        assert_eq!(at("mm.input_norm.weight"), (vec![4608], GgmlType::F32));
        assert_eq!(
            at("v.vit_merger.ds_ffn_up.weight"),
            (vec![4608, 17216], GgmlType::BF16)
        );
        assert_eq!(
            at("v.vit_merger.ds_ffn_up.bias"),
            (vec![17216], GgmlType::F32)
        );
        assert_eq!(
            at("v.vit_merger.ds_ffn_down.weight"),
            (vec![17216, 1152], GgmlType::BF16)
        );
        assert_eq!(at("v.vit_merger.ds_ln.weight"), (vec![4608], GgmlType::F32));
        assert_eq!(
            at("v.vit_merger.attn_out.weight"),
            (vec![1152, 1152], GgmlType::BF16)
        );
    }

    /// A file that is the table passes; the changes a converter or a wrong file would make are
    /// refused by the tensor's name.
    #[test]
    fn the_check_refuses_by_name() {
        let hp = hp();
        changes_are_refused_by_name(
            super::super::PROJECTOR_TYPE,
            &expected(&hp, Export::Bf16),
            460,
            "v.vit_merger.attn_v.weight",
            |l| text(check(&hp, triples(l))),
        );
    }
}
