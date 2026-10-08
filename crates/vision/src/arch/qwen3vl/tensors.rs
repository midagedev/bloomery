//! The tensors a `qwen3vl_merger` file must hold: every name of [`super::names`] once, at the
//! shape the hyperparameters give it and the type this crate reads, and nothing else.
//!
//! The file is an f16 or a bf16 export of llama.cpp's converter. Matrices are f16 in the first and
//! bf16 in the second; the two patch kernels are f16 in an f16 export and f32 in any other (the
//! converter keeps that tensor out of the narrow type); biases, norm gains and the position table
//! are f32. The export is read from the type of the merger's first weight, `mm.0.weight`, and every
//! other tensor must then hold the type that export gives it.

use gguf::GgmlType;

use super::{Hparams, names};
use crate::VisionError;
use crate::arch::table;

/// Rows of the learned position table: a 48×48 grid.
const POSITIONS: u64 = 2304;

/// One tensor the file must hold, dims in ggml `ne[]` order (the contiguous axis first). The
/// file's tensors all load the same way, so a row carries no note on where it lives.
pub type Expected = table::Expected<()>;

/// How the converter typed the file's matrices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Export {
    F16,
    Bf16,
}

impl Export {
    /// The export a file is, from the type of its `mm.0.weight`; `None` for a type no export of
    /// this crate's has.
    #[must_use]
    pub fn of(merger_weight: GgmlType) -> Option<Export> {
        match merger_weight {
            GgmlType::F16 => Some(Export::F16),
            GgmlType::BF16 => Some(Export::Bf16),
            _ => None,
        }
    }

    fn matrix(self) -> GgmlType {
        match self {
            Export::F16 => GgmlType::F16,
            Export::Bf16 => GgmlType::BF16,
        }
    }

    fn patch(self) -> GgmlType {
        match self {
            Export::F16 => GgmlType::F16,
            Export::Bf16 => GgmlType::F32,
        }
    }
}

/// Every tensor of an `export` file with these hyperparameters, in the table's order.
#[must_use]
pub fn expected(hp: &Hparams, export: Export) -> Vec<Expected> {
    let d = |n: usize| n as u64;
    let (dim, ff, p, out) = (d(hp.dim), d(hp.ff), d(hp.patch), d(hp.out_dim));
    let merged = dim * d(hp.merge * hp.merge);
    let mat = |name: String, dims: &[u64]| Expected::row(name, dims, export.matrix(), ());
    let vec = |name: String, n: u64| Expected::row(name, &[n], GgmlType::F32, ());
    let patch = |name: String| Expected::row(name, &[p, p, 3, dim], export.patch(), ());
    let mut all = vec![
        patch(names::patch_embd_weight()),
        patch(names::patch_embd_weight_second_frame()),
        vec(names::patch_embd_bias(), dim),
        Expected::row(names::position_embd(), &[dim, POSITIONS], GgmlType::F32, ()),
    ];
    for b in 0..hp.n_layer {
        all.extend([
            vec(names::ln1_weight(b), dim),
            vec(names::ln1_bias(b), dim),
            mat(names::attn_qkv_weight(b), &[dim, 3 * dim]),
            vec(names::attn_qkv_bias(b), 3 * dim),
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
        vec(names::post_ln_weight(), dim),
        vec(names::post_ln_bias(), dim),
        mat(names::mm0_weight(), &[merged, merged]),
        vec(names::mm0_bias(), merged),
        mat(names::mm2_weight(), &[merged, out]),
        vec(names::mm2_bias(), out),
    ]);
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
    let merger = names::mm0_weight();
    let err = |detail: String| VisionError::Tensor {
        name: merger.clone(),
        detail,
    };
    let Some(&(_, _, ty)) = tensors.iter().find(|(name, ..)| *name == merger) else {
        return Err(err("is missing from the file".into()));
    };
    let Some(export) = Export::of(ty) else {
        return Err(err(format!(
            "is {ty}; the matrices of a {} file are f16 or bf16",
            super::PROJECTOR_TYPE
        )));
    };
    table::check(super::PROJECTOR_TYPE, &expected(hp, export), tensors)
}

#[cfg(test)]
mod tests {
    use gguf::GgmlType;

    use super::{Export, check, expected};
    use crate::arch::qwen3vl::Hparams;
    use crate::arch::table::testing::{changes_are_refused_by_name, list, row_of, text, triples};

    /// Clef Flash's file.
    fn hp() -> Hparams {
        Hparams {
            n_layer: 27,
            dim: 1152,
            n_head: 16,
            ff: 4304,
            patch: 16,
            merge: 2,
            out_dim: 4096,
            eps: 1e-6,
        }
    }

    /// 12 tensors per block, 4 before and 6 after: the 334 of the file; 112 of them f16 and 222
    /// f32 in the f16 export, 918,145,984 bytes of tensor data (the header's last tensor ends at
    /// that offset).
    #[test]
    fn table_is_334() {
        let t = expected(&hp(), Export::F16);
        assert_eq!(t.len(), 27 * 12 + 10);
        let of = |ty| t.iter().filter(|e| e.ty == ty).count();
        assert_eq!((of(GgmlType::F16), of(GgmlType::F32)), (112, 222));
        assert_eq!(
            t.iter().map(super::Expected::bytes).sum::<u64>(),
            918_145_984
        );
    }

    /// A bf16 export differs in the type of 110 matrices and of the two patch kernels, and its
    /// data is 3,538,944 bytes (the two kernels' extra two bytes a value) past the f16 export's.
    #[test]
    fn the_bf16_export_types_the_kernels_f32() {
        let t = expected(&hp(), Export::Bf16);
        assert_eq!(t.len(), 334);
        let of = |ty| t.iter().filter(|e| e.ty == ty).count();
        assert_eq!((of(GgmlType::BF16), of(GgmlType::F32)), (110, 224));
        assert_eq!(
            t.iter().map(super::Expected::bytes).sum::<u64>(),
            918_145_984 + 3_538_944
        );
    }

    /// The table itself passes in both exports; one renamed tensor is refused by its new name, a
    /// dropped one is named as missing, a doubled one as twice, a reshaped one by its dims, a
    /// retyped one by its type.
    #[test]
    fn a_changed_tensor_list_is_refused_by_name() {
        let hp = hp();
        for export in [Export::F16, Export::Bf16] {
            changes_are_refused_by_name(
                super::super::PROJECTOR_TYPE,
                &expected(&hp, export),
                334,
                "mm.2.bias",
                |l| text(check(&hp, triples(l))),
            );
        }
    }

    /// An extra tensor (a 28th block) is refused by its name; so is a block missing from the
    /// middle.
    #[test]
    fn an_extra_or_missing_block_tensor_is_refused_by_name() {
        let hp = hp();
        let t = expected(&hp, Export::F16);
        let mut extra = list(&t);
        extra.push(("v.blk.27.ln1.weight".into(), vec![1152], GgmlType::F32));
        assert_eq!(
            text(check(&hp, triples(&extra))).unwrap_err(),
            "tensor v.blk.27.ln1.weight: is not a qwen3vl_merger tensor name"
        );
        let mut deepstack = list(&t);
        deepstack.push((
            "v.deepstack.0.norm.weight".into(),
            vec![4608],
            GgmlType::F32,
        ));
        assert_eq!(
            text(check(&hp, triples(&deepstack))).unwrap_err(),
            "tensor v.deepstack.0.norm.weight: is not a qwen3vl_merger tensor name"
        );
        let mut gap = list(&t);
        let at = gap
            .iter()
            .position(|r| r.0 == "v.blk.13.ffn_up.bias")
            .expect("row");
        gap.remove(at);
        assert_eq!(
            text(check(&hp, triples(&gap))).unwrap_err(),
            "tensor v.blk.13.ffn_up.bias: is missing from the file"
        );
    }

    /// The export is read from `mm.0.weight`: a type no export has, or no such tensor, is refused
    /// by its name; one matrix of the other export's type is refused by its own.
    #[test]
    fn the_export_is_the_mergers_and_every_tensor_follows_it() {
        let hp = hp();
        let t = expected(&hp, Export::F16);
        let mut odd = list(&t);
        let mm0 = odd.iter().position(|r| r.0 == "mm.0.weight").expect("row");
        odd[mm0].2 = GgmlType::Q8_0;
        assert_eq!(
            text(check(&hp, triples(&odd))).unwrap_err(),
            "tensor mm.0.weight: is q8_0; the matrices of a qwen3vl_merger file are f16 or bf16"
        );
        let mut none = list(&t);
        none.remove(mm0);
        assert_eq!(
            text(check(&hp, triples(&none))).unwrap_err(),
            "tensor mm.0.weight: is missing from the file"
        );
        let mut mixed = list(&t);
        let qkv = mixed
            .iter()
            .position(|r| r.0 == "v.blk.3.attn_qkv.weight")
            .expect("row");
        mixed[qkv].2 = GgmlType::BF16;
        assert_eq!(
            text(check(&hp, triples(&mixed))).unwrap_err(),
            "tensor v.blk.3.attn_qkv.weight: is bf16; this crate reads it as f16"
        );
        // The merger's weight names the export, so a bf16 file without it is refused for that
        // and not for the types of the tensors it does hold.
        let b = expected(&hp, Export::Bf16);
        let mut none_b = list(&b);
        let at = none_b
            .iter()
            .position(|r| r.0 == "mm.0.weight")
            .expect("row");
        none_b.remove(at);
        assert_eq!(
            text(check(&hp, triples(&none_b))).unwrap_err(),
            "tensor mm.0.weight: is missing from the file"
        );
        // A bf16 export's kernels are f32; an f16 kernel in it is refused.
        let mut kernel = list(&b);
        let k = row_of(&kernel, "v.patch_embd.weight.1");
        kernel[k].2 = GgmlType::F16;
        assert_eq!(
            text(check(&hp, triples(&kernel))).unwrap_err(),
            "tensor v.patch_embd.weight.1: is f16; this crate reads it as f32"
        );
    }
}
