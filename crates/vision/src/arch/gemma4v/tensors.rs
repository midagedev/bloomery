//! The vision tensors of a multimodal Gemma4 projector; audio tensors stay on the host.

use super::{Hparams, PROJECTOR_TYPE};
use crate::{
    VisionError,
    arch::table::{self, Expected},
};
use gguf::{GgmlType, Gguf};

/// Names and ggml shapes of every required vision tensor.
#[must_use]
pub fn expected(hp: &Hparams) -> Vec<Expected<()>> {
    let (d, f, out, p) = (
        hp.dim as u64,
        hp.ff as u64,
        hp.out_dim as u64,
        hp.patch as u64,
    );
    let mut rows = vec![
        Expected::row(
            "v.patch_embd.weight".into(),
            &[p, p, 3, d],
            GgmlType::F32,
            (),
        ),
        Expected::row(
            "v.position_embd.weight".into(),
            &[d, hp.pos_rows as u64, 2],
            GgmlType::F32,
            (),
        ),
        Expected::row(
            "mm.input_projection.weight".into(),
            &[d, out],
            GgmlType::BF16,
            (),
        ),
    ];
    for l in 0..hp.n_layer {
        for stem in [
            "ln1",
            "ln2",
            "attn_post_norm",
            "ffn_post_norm",
            "attn_q_norm",
            "attn_k_norm",
        ] {
            let n = if stem.ends_with("_q_norm") || stem.ends_with("_k_norm") {
                64
            } else {
                d
            };
            rows.push(Expected::row(
                format!("v.blk.{l}.{stem}.weight"),
                &[n],
                GgmlType::F32,
                (),
            ));
        }
        for (stem, dims) in [
            ("attn_q", [d, d]),
            ("attn_k", [d, d]),
            ("attn_v", [d, d]),
            ("attn_out", [d, d]),
            ("ffn_gate", [d, f]),
            ("ffn_up", [d, f]),
            ("ffn_down", [f, d]),
        ] {
            rows.push(Expected::row(
                format!("v.blk.{l}.{stem}.weight"),
                &dims,
                GgmlType::BF16,
                (),
            ));
        }
    }
    rows
}

/// Check the complete vision half, including optional standardization and clamp scalars.
pub fn check(file: &Gguf, hp: &Hparams) -> Result<Vec<Expected<()>>, VisionError> {
    let mut rows = expected(hp);
    let has_bias = file.find("v.std_bias").is_some();
    let has_scale = file.find("v.std_scale").is_some();
    if has_bias != has_scale {
        return Err(VisionError::Tensor {
            name: "v.std_bias/v.std_scale".into(),
            detail: "standardization needs both vectors".into(),
        });
    }
    if has_bias {
        for name in ["v.std_bias", "v.std_scale"] {
            rows.push(Expected::row(
                name.into(),
                &[hp.dim as u64],
                GgmlType::F32,
                (),
            ));
        }
    }
    let matrices: Vec<String> = rows
        .iter()
        .filter(|r| r.dims.len() == 2)
        .map(|r| r.name.clone())
        .collect();
    for name in matrices {
        for suffix in ["input_min", "input_max", "output_min", "output_max"] {
            let n = name.replace(".weight", &format!(".{suffix}"));
            if file.find(&n).is_some() {
                rows.push(Expected::row(n, &[1], GgmlType::F32, ()));
            }
        }
    }
    table::check(
        PROJECTOR_TYPE,
        &rows,
        file.iter_tensors()
            .filter(|t| !t.name.starts_with("a.") && !t.name.starts_with("mm.a."))
            .map(|t| (t.name.as_str(), t.dims.as_slice(), t.ty)),
    )?;
    Ok(rows)
}
