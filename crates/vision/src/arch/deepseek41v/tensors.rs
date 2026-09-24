//! The tensors a `deepseek41v` file must hold: every name of [`super::names`] once, at the shape
//! the hyperparameters give it and the type this crate reads, and nothing else.
//!
//! Matrices are bf16 (the checkpoint's own type — the file is a relabelling of it); gains, biases
//! and delimiter rows are f32. A tensor under a name the table does not know is refused by that
//! name before anything else is looked at, so a file of another layout fails on its first
//! unfamiliar tensor, not on a shape.

use std::collections::{HashMap, HashSet};

use gguf::GgmlType;

use super::{Hparams, names};
use crate::VisionError;

/// One tensor the file must hold, dims in ggml `ne[]` order (the contiguous axis first).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expected {
    pub name: String,
    pub dims: Vec<u64>,
    pub ty: GgmlType,
}

/// Every tensor of a file with these hyperparameters, in the table's order.
#[must_use]
pub fn expected(hp: &Hparams) -> Vec<Expected> {
    let d = |n: usize| n as u64;
    let (dim, ff, p, out) = (d(hp.dim), d(hp.ff), d(hp.patch), d(hp.out_dim));
    let unfold = dim * d(hp.downsample * hp.downsample);
    let mat = |name: String, dims: &[u64]| Expected {
        name,
        dims: dims.to_vec(),
        ty: GgmlType::BF16,
    };
    let vec = |name: String, n: u64| Expected {
        name,
        dims: vec![n],
        ty: GgmlType::F32,
    };
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
        vec(names::img_start(), out),
        vec(names::img_end(), out),
        vec(names::image_newline(), out),
    ]);
    all
}

/// Check a file's tensors (name, ggml dims, type) against [`expected`]: every one known, none
/// twice, each at its shape and type, and none of the table missing. Returns how many were named.
pub fn check<'a>(
    hp: &Hparams,
    tensors: impl IntoIterator<Item = (&'a str, &'a [u64], GgmlType)>,
) -> Result<usize, VisionError> {
    let table = expected(hp);
    let by_name: HashMap<&str, &Expected> = table.iter().map(|e| (e.name.as_str(), e)).collect();
    let mut seen: HashSet<&str> = HashSet::with_capacity(table.len());
    let err = |name: &str, detail: String| VisionError::Tensor {
        name: name.to_string(),
        detail,
    };
    for (name, dims, ty) in tensors {
        let want = by_name.get(name).ok_or_else(|| {
            err(
                name,
                format!("is not a {} tensor name", super::PROJECTOR_TYPE),
            )
        })?;
        if !seen.insert(want.name.as_str()) {
            return Err(err(name, "appears twice".into()));
        }
        if dims != want.dims.as_slice() {
            return Err(err(
                name,
                format!("has dims {dims:?}; the table gives {:?}", want.dims),
            ));
        }
        if ty != want.ty {
            return Err(err(
                name,
                format!("is {ty}; this crate reads it as {}", want.ty),
            ));
        }
    }
    if let Some(missing) = table.iter().find(|e| !seen.contains(e.name.as_str())) {
        return Err(err(&missing.name, "is missing from the file".into()));
    }
    Ok(seen.len())
}

#[cfg(test)]
mod tests {
    use super::{check, expected};
    use crate::arch::deepseek41v::Hparams;

    fn hp() -> Hparams {
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
        let t = expected(&hp);
        let list = |t: &[super::Expected]| {
            t.iter()
                .map(|e| (e.name.clone(), e.dims.clone(), e.ty))
                .collect::<Vec<_>>()
        };
        let run = |l: &[(String, Vec<u64>, gguf::GgmlType)]| {
            check(
                &hp,
                l.iter().map(|(n, d, ty)| (n.as_str(), d.as_slice(), *ty)),
            )
            .map_err(|e| e.to_string())
        };
        assert_eq!(run(&list(&t)), Ok(298));

        let mut renamed = list(&t);
        renamed[5].0 = "v.blk.0.attn_o.weight".into();
        assert_eq!(
            run(&renamed).unwrap_err(),
            "tensor v.blk.0.attn_o.weight: is not a deepseek41v tensor name"
        );
        let mut dropped = list(&t);
        dropped.pop();
        assert_eq!(
            run(&dropped).unwrap_err(),
            "tensor v.image_newline: is missing from the file"
        );
        let mut doubled = list(&t);
        doubled.push(doubled[0].clone());
        assert_eq!(
            run(&doubled).unwrap_err(),
            "tensor v.patch_embd.weight: appears twice"
        );
        let mut reshaped = list(&t);
        reshaped[2].1 = vec![3072, 1024];
        assert!(
            run(&reshaped)
                .unwrap_err()
                .starts_with("tensor v.blk.0.ln1.weight: has dims [3072, 1024]")
        );
    }
}
