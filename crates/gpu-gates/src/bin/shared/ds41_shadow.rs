//! The kernels the V4.1 step launches for a projection, as the file's tensor
//! types decide them: the step and skew gates' shadow tables.

use bloomery_gpu_gates::GateError;
use gguf::{GgmlType, Split};
use model::arch::deepseek41::names;

/// Tensor `name`'s type in `split`.
fn type_of(split: &Split, name: &str) -> Result<GgmlType, GateError> {
    split
        .find(name)
        .map(|(_, t)| t.ty)
        .ok_or_else(|| format!("{name} is not in the file").into())
}

/// The kernels the step launches for one dense projection of file type `ty`,
/// in stream order: the q8_1 form of its input first where a K-quant gemv
/// reads one and the step makes it for this projection alone.
fn projection(ty: GgmlType, name: &str) -> Result<&'static [&'static str], GateError> {
    match ty {
        GgmlType::Q8_0 => Ok(&["q8_0_gemv"]),
        GgmlType::Q3_K => Ok(&["q3k_quantize_q8_1", "q3k_gemv"]),
        GgmlType::Q4_K => Ok(&["q3k_quantize_q8_1", "q4k_gemv"]),
        GgmlType::Q5_K => Ok(&["ds41_q5k_gemv_f32"]),
        other => Err(format!("{name}: {other} has no dense kernel").into()),
    }
}

/// The MoE piece's own work in layer `l`'s host-leg shadow, in stream order:
/// HC_PRE, with card experts the routed gate·up, h's q8_1 and the routed
/// down, then the shared expert's gate·up (`ds41_shexp_gate_up`, or its
/// q3_K twin when both weights are q3_K) and down ([`projection`]).
pub fn shadow_kernels(
    split: &Split,
    l: usize,
    card_experts: bool,
) -> Result<Vec<&'static str>, GateError> {
    let mut k = vec!["ds41_hc_pre"];
    if card_experts {
        k.extend(["ds41_expert_gate_up", "q3k_quantize_q8_1", "q4k_gemv_sel"]);
    }
    let gate = type_of(split, &names::ffn_gate_shexp(l))?;
    let up = type_of(split, &names::ffn_up_shexp(l))?;
    k.push(if (gate, up) == (GgmlType::Q3_K, GgmlType::Q3_K) {
        "ds41_shexp_gate_up_q3k"
    } else {
        "ds41_shexp_gate_up"
    });
    let down = names::ffn_down_shexp(l);
    k.extend(projection(type_of(split, &down)?, &down)?);
    Ok(k)
}

/// Engram site `l`'s token-only work, in stream order: its rows decoded
/// (`ds41_glue_engram_rows`, or `_q3k` for a q3_K table), `engram_wkv`
/// ([`projection`]) and the key norm. It reads the step image and weights
/// alone.
pub fn engram_kv_kernels(split: &Split, l: usize) -> Result<Vec<&'static str>, GateError> {
    let table = names::engram_embd(l);
    let mut k = vec![match type_of(split, &table)? {
        GgmlType::Q8_0 => "ds41_glue_engram_rows",
        GgmlType::Q3_K => "ds41_glue_engram_rows_q3k",
        other => return Err(format!("{table}: {other} has no row kernel").into()),
    }];
    let wkv = names::engram_wkv(l);
    k.extend(projection(type_of(split, &wkv)?, &wkv)?);
    k.push("ds41_engram_key_norm");
    Ok(k)
}
