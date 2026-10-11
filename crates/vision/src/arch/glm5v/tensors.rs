//! The tensors a `glm5v` file must hold: every name of [`super::names`] once, at the shape the
//! hyperparameters give it and the type this crate reads, and nothing else.
//!
//! The file is a bf16 export of llama.cpp's converter: matrices are bf16; the two patch kernels
//! and the merger's convolution are f32 (the converter keeps convolution weights out of the
//! narrow type), narrowed to bf16 on upload ([`OnCard::Bf16`]); biases and norm gains are f32. A
//! tensor of any other type, as in an f16 export, is refused by its name and type.

use gguf::GgmlType;

use super::{Hparams, MERGER_FF, names};
use crate::VisionError;
use crate::arch::table;

/// How the card holds a tensor of the file, which its bytes depend on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnCard {
    /// As the file stores it.
    File,
    /// Narrowed to bf16 on upload (`arch::narrow`: refused unless every value round-trips).
    Bf16,
}

/// One tensor the file must hold, dims in ggml `ne[]` order (the contiguous axis first), and how
/// the card holds it.
pub type Expected = table::Expected<OnCard>;

/// Every tensor of the file with these hyperparameters, in the table's order.
#[must_use]
pub fn expected(hp: &Hparams) -> Vec<Expected> {
    let d = |n: usize| n as u64;
    let (dim, ff, p, out) = (d(hp.dim), d(hp.ff), d(hp.patch), d(hp.out_dim));
    let (head, merge, mff) = (d(hp.dim / hp.n_head), d(hp.merge), d(MERGER_FF));
    let mat = |name: String, dims: &[u64]| Expected::row(name, dims, GgmlType::BF16, OnCard::File);
    let vec = |name: String, n: u64| Expected::row(name, &[n], GgmlType::F32, OnCard::File);
    let narrowed =
        |name: String, dims: &[u64]| Expected::row(name, dims, GgmlType::F32, OnCard::Bf16);
    let mut all = vec![
        narrowed(names::patch_embd_weight(), &[p, p, 3, dim]),
        narrowed(names::patch_embd_weight_second_frame(), &[p, p, 3, dim]),
        vec(names::patch_embd_bias(), dim),
    ];
    for b in 0..hp.n_layer {
        all.extend([
            vec(names::ln1_weight(b), dim),
            vec(names::attn_q_norm_weight(b), head),
            vec(names::attn_k_norm_weight(b), head),
            mat(names::attn_qkv_weight(b), &[dim, 3 * dim]),
            vec(names::attn_qkv_bias(b), 3 * dim),
            mat(names::attn_out_weight(b), &[dim, dim]),
            vec(names::attn_out_bias(b), dim),
            vec(names::ln2_weight(b), dim),
            mat(names::ffn_gate_weight(b), &[dim, ff]),
            vec(names::ffn_gate_bias(b), ff),
            mat(names::ffn_up_weight(b), &[dim, ff]),
            vec(names::ffn_up_bias(b), ff),
            mat(names::ffn_down_weight(b), &[ff, dim]),
            vec(names::ffn_down_bias(b), dim),
        ]);
    }
    all.extend([
        vec(names::post_ln_weight(), dim),
        narrowed(names::patch_merger_weight(), &[merge, merge, dim, out]),
        vec(names::patch_merger_bias(), out),
        mat(names::fc_weight(), &[out, out]),
        vec(names::post_norm_weight(), out),
        vec(names::post_norm_bias(), out),
        mat(names::mm_gate_weight(), &[out, mff]),
        mat(names::mm_up_weight(), &[out, mff]),
        mat(names::mm_down_weight(), &[mff, out]),
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

    use super::{OnCard, check, expected};
    use crate::arch::glm5v::hparams::tests::hp;
    use crate::arch::table::testing::{changes_are_refused_by_name, list, text, triples};

    /// 14 tensors per block and 9 after the blocks and 3 before: the 348 of the file, 124 bf16
    /// and 224 f32, 1,163,988,992 bytes of tensor data (the header's last tensor,
    /// `v.post_ln.weight`, ends at 1,163,984,896 + 4,096).
    #[test]
    fn table_is_348() {
        let t = expected(&hp());
        assert_eq!(t.len(), 24 * 14 + 12);
        let of = |ty| t.iter().filter(|e| e.ty == ty).count();
        assert_eq!((of(GgmlType::BF16), of(GgmlType::F32)), (124, 224));
        assert_eq!(
            t.iter().map(super::Expected::bytes).sum::<u64>(),
            1_163_988_992
        );
        let narrowed: Vec<&str> = t
            .iter()
            .filter(|e| e.home == OnCard::Bf16)
            .map(|e| e.name.as_str())
            .collect();
        assert_eq!(
            narrowed,
            [
                "v.patch_embd.weight",
                "v.patch_embd.weight.1",
                "mm.patch_merger.weight"
            ]
        );
    }

    /// The table itself passes; one renamed tensor is refused by its new name, a dropped one is
    /// named as missing, a doubled one as twice, a reshaped one by its dims, a retyped one by its
    /// type.
    #[test]
    fn a_changed_tensor_list_is_refused_by_name() {
        let hp = hp();
        changes_are_refused_by_name(
            super::super::PROJECTOR_TYPE,
            &expected(&hp),
            348,
            "mm.down.weight",
            |l| text(check(&hp, triples(l))),
        );
    }

    /// A tensor of another layout is refused by its name: a 25th block, a position table, the
    /// merger MLP's biases (llama.cpp reads them when present; this file has none), and a block
    /// without its query norm is named as missing.
    #[test]
    fn an_extra_or_missing_tensor_is_refused_by_name() {
        let hp = hp();
        let t = expected(&hp);
        for extra in [
            "v.blk.24.ln1.weight",
            "v.position_embd.weight",
            "mm.up.bias",
            "v.blk.0.ln1.bias",
        ] {
            let mut l = list(&t);
            l.push((extra.into(), vec![1024], GgmlType::F32));
            assert_eq!(
                text(check(&hp, triples(&l))).unwrap_err(),
                format!("tensor {extra}: is not a glm5v tensor name")
            );
        }
        let mut gap = list(&t);
        let at = gap
            .iter()
            .position(|r| r.0 == "v.blk.13.attn_q_norm.weight")
            .expect("row");
        gap.remove(at);
        assert_eq!(
            text(check(&hp, triples(&gap))).unwrap_err(),
            "tensor v.blk.13.attn_q_norm.weight: is missing from the file"
        );
    }

    /// The shapes the merger and the per-head norms take: the convolution is
    /// `[merge, merge, dim, out]`, the MLP's gate and up `[out, 10240]`, its down `[10240, out]`,
    /// and a head norm is one head wide; a reshaped one is refused with the dims it holds.
    #[test]
    fn the_merger_and_head_norm_shapes_are_pinned() {
        let hp = hp();
        let t = expected(&hp);
        let dims = |n: &str| t.iter().find(|e| e.name == n).expect("row").dims.clone();
        assert_eq!(dims("mm.patch_merger.weight"), [2, 2, 1024, 4096]);
        assert_eq!(dims("mm.model.fc.weight"), [4096, 4096]);
        assert_eq!(dims("mm.gate.weight"), [4096, 10_240]);
        assert_eq!(dims("mm.up.weight"), [4096, 10_240]);
        assert_eq!(dims("mm.down.weight"), [10_240, 4096]);
        assert_eq!(dims("v.blk.5.attn_q_norm.weight"), [64]);
        let mut wrong = list(&t);
        let at = wrong
            .iter()
            .position(|r| r.0 == "mm.gate.weight")
            .expect("row");
        wrong[at].1 = vec![4096, 8192];
        assert_eq!(
            text(check(&hp, triples(&wrong))).unwrap_err(),
            "tensor mm.gate.weight: has dims [4096, 8192]; the table gives [4096, 10240]"
        );
    }

    /// A convolution weight of another type is refused by name and type.
    #[test]
    fn an_f16_convolution_is_refused() {
        let hp = hp();
        let mut l = list(&expected(&hp));
        let at = l
            .iter()
            .position(|r| r.0 == "mm.patch_merger.weight")
            .expect("row");
        l[at].2 = GgmlType::F16;
        assert_eq!(
            text(check(&hp, triples(&l))).unwrap_err(),
            "tensor mm.patch_merger.weight: is f16; this crate reads it as f32"
        );
    }
}
