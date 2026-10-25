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
/// reads one and the step makes it for this projection alone. `prefetch` is
/// the device's `Gpu::row_prefetch`, which names the Q4_K gemvs' entries.
fn projection(
    ty: GgmlType,
    name: &str,
    prefetch: bool,
) -> Result<&'static [&'static str], GateError> {
    match ty {
        GgmlType::Q8_0 => Ok(&["q8_0_gemv"]),
        GgmlType::Q3_K => Ok(&["q3k_quantize_q8_1", "q3k_gemv"]),
        GgmlType::Q4_K if prefetch => Ok(&["q3k_quantize_q8_1", "q4k_gemv_prefetch"]),
        GgmlType::Q4_K => Ok(&["q3k_quantize_q8_1", "q4k_gemv"]),
        GgmlType::Q5_K => Ok(&["ds41_q5k_gemv_f32"]),
        other => Err(format!("{name}: {other} has no dense kernel").into()),
    }
}

/// The HC_PRE kernel that reads hyper-connection fn `name`: the q3_K split-K
/// walk's, or the f32 table's.
pub fn hc_pre_kernel(split: &Split, name: &str) -> Result<&'static str, GateError> {
    match type_of(split, name)? {
        GgmlType::Q3_K => Ok("ds41_hc_pre"),
        GgmlType::F32 => Ok("ds41_hc_pre_f32"),
        other => Err(format!("{name}: {other} has no hyper-connection kernel").into()),
    }
}

/// The kernel that runs layer `l`'s routed gate·up of file type `ty` on the
/// card: the fused layer's Q3_K entry, or the common leg's entry for the type
/// (`kernels::gate_up`).
fn routed_gate_up(ty: GgmlType, l: usize) -> Result<&'static str, GateError> {
    Ok(match ty {
        GgmlType::Q3_K => "ds41_expert_gate_up",
        GgmlType::IQ2_XXS => "iq2_xxs_gate_up_sel",
        GgmlType::IQ2_XS => "iq2_xs_gate_up_sel",
        GgmlType::IQ2_S => "iq2_s_gate_up_sel",
        GgmlType::IQ1_S => "iq1_s_gate_up_sel",
        GgmlType::IQ1_M => "iq1_m_gate_up_sel",
        GgmlType::IQ3_XXS => "iq3_xxs_gate_up_sel",
        GgmlType::IQ3_S => "iq3_s_gate_up_sel",
        GgmlType::IQ4_XS => "iq4_xs_gate_up_sel",
        other => {
            return Err(
                format!("layer {l}: {other} routed gate·up has no card kernel name here").into(),
            );
        }
    })
}

/// The kernel that runs layer `l`'s routed down of file type `ty` on the card.
fn routed_down(ty: GgmlType, prefetch: bool, l: usize) -> Result<&'static str, GateError> {
    Ok(match ty {
        GgmlType::Q4_K if prefetch => "q4k_gemv_sel_prefetch",
        GgmlType::Q4_K => "q4k_gemv_sel",
        GgmlType::IQ2_XXS => "iq2_xxs_gemv_sel",
        GgmlType::IQ2_XS => "iq2_xs_gemv_sel",
        GgmlType::IQ2_S => "iq2_s_gemv_sel",
        GgmlType::IQ1_S => "iq1_s_gemv_sel",
        GgmlType::IQ1_M => "iq1_m_gemv_sel",
        GgmlType::IQ3_XXS => "iq3_xxs_gemv_sel",
        GgmlType::IQ3_S => "iq3_s_gemv_sel",
        GgmlType::IQ4_XS => "iq4_xs_gemv_sel",
        other => {
            return Err(
                format!("layer {l}: {other} routed down has no card kernel name here").into(),
            );
        }
    })
}

/// The MoE piece's own work in layer `l`'s host-leg shadow, in stream order:
/// HC_PRE ([`hc_pre_kernel`]), with card experts the routed gate·up, the q8_1
/// of h's card-slot columns and the routed down (each by the stack's file
/// type), then the shared expert's
/// gate·up (`ds41_shexp_gate_up`, or its q3_K or q5_K twin when both weights
/// are of that type) and down ([`projection`]).
pub fn shadow_kernels(
    split: &Split,
    l: usize,
    card_experts: bool,
    prefetch: bool,
) -> Result<Vec<&'static str>, GateError> {
    let mut k = vec![hc_pre_kernel(split, &names::hc_ffn_fn(l))?];
    if card_experts {
        // PIN(2026-09-26, fixup6): q3k_quantize_q8_1 → q8_1_quantize_sel, same launch count.
        // h's q8_1 covers the card slots' columns alone: a column the host serves
        // holds what the last faulted step left there, a reset keeps it, and a
        // quantizer that read it raised quant_column on a clean step
        // (gate_deepseek41_prefill, "fault-reset steps").
        let gate_up = type_of(split, &names::ffn_gate_exps(l))?;
        let down = type_of(split, &names::ffn_down_exps(l))?;
        k.extend([
            routed_gate_up(gate_up, l)?,
            "q8_1_quantize_sel",
            routed_down(down, prefetch, l)?,
        ]);
    }
    let gate = type_of(split, &names::ffn_gate_shexp(l))?;
    let up = type_of(split, &names::ffn_up_shexp(l))?;
    k.push(match (gate, up) {
        (GgmlType::Q3_K, GgmlType::Q3_K) => "ds41_shexp_gate_up_q3k",
        (GgmlType::Q5_K, GgmlType::Q5_K) => "ds41_shexp_gate_up_q5k",
        _ => "ds41_shexp_gate_up",
    });
    let down = names::ffn_down_shexp(l);
    k.extend(projection(type_of(split, &down)?, &down, prefetch)?);
    Ok(k)
}

/// Engram site `l`'s token-only work, in stream order: its rows decoded
/// (`ds41_glue_engram_rows`, or `_q3k` for a q3_K table, `_iq4nl` for an
/// iq4_nl one), `engram_wkv`
/// ([`projection`]) and the key norm. It reads the step image and weights
/// alone.
pub fn engram_kv_kernels(
    split: &Split,
    l: usize,
    prefetch: bool,
) -> Result<Vec<&'static str>, GateError> {
    let table = names::engram_embd(l);
    let mut k = vec![match type_of(split, &table)? {
        GgmlType::Q8_0 => "ds41_glue_engram_rows",
        GgmlType::Q3_K => "ds41_glue_engram_rows_q3k",
        GgmlType::IQ4_NL => "ds41_glue_engram_rows_iq4nl",
        other => return Err(format!("{table}: {other} has no row kernel").into()),
    }];
    let wkv = names::engram_wkv(l);
    k.extend(projection(type_of(split, &wkv)?, &wkv, prefetch)?);
    k.push("ds41_engram_key_norm");
    Ok(k)
}
