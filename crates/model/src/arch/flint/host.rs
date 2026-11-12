//! Flint's validated routed block as the common host and NVMe expert tiers serve it.

use models::{Act, Ffn, ModelSpec};

use crate::ModelError;
use crate::arch::names;
use crate::moe::{HostLayer, HostLayerSpec};
use crate::ops::ExpertInput;
use crate::r8file::R8Source;

/// Build the requested layer's routed stacks with its description's SwiGLU (plain or the
/// gate-first clamp, [`HostLayer::set_act`]) and activated-value scaling. Dense layers, layers
/// outside the description and routed blocks of another activation are refused before a tensor is
/// opened. Load-time only.
pub fn layer(src: R8Source<'_>, spec: &ModelSpec, l: usize) -> Result<HostLayer, ModelError> {
    let refuse = |detail: String| ModelError::Metadata {
        key: format!("layer {l} routed block"),
        detail,
    };
    let block = spec
        .layers
        .get(l)
        .ok_or_else(|| refuse(format!("outside the file's {} layers", spec.layers.len())))?;
    let Ffn::Moe(moe) = &block.ffn else {
        return Err(refuse("is dense, not a routed block".to_string()));
    };
    match moe.act {
        Act::SwiGlu { .. } | Act::SwiGluGateFirst { .. } => {}
        other => {
            return Err(refuse(format!(
                "activation {other:?}; this host runs SwiGLU and the gate-first clamp"
            )));
        }
    }
    let [gate, up, down] = names::routed_stacks(l);
    let mut layer = HostLayer::build(
        src,
        &HostLayerSpec {
            gate: &gate,
            up: &up,
            down: &down,
            n_expert: moe.experts as usize,
            embd: spec.hidden as usize,
            ff: moe.expert_ff as usize,
            swiglu_limit: 0.0,
        },
    )?;
    layer.set_act(moe.act)?;
    layer.set_expert_input(ExpertInput::AmaxF16);
    Ok(layer)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{keys, tensors};
    use super::*;
    use crate::arch::synthetic::{Ty, V, header_shaped, header_typed};
    use crate::ops::ExpertForm;

    #[test]
    fn refuses_dense_outside_and_non_swiglu_before_opening_experts() {
        let path = header_shaped(
            "flint-host-layer",
            "flint",
            &keys(),
            &[("tokenizer.ggml.pre", V::Str("flint"))],
            &tensors(),
        );
        let split = gguf::Split::open(&path).unwrap();
        let mut spec = super::super::spec::read(&split).unwrap().spec;
        let src = R8Source::rows(&split);
        for (l, fragment) in [(0, "is dense"), (4, "outside the file's 4 layers")] {
            let error = layer(src, &spec, l).err().expect("typed refusal");
            assert!(matches!(error, ModelError::Metadata { ref key, ref detail }
                if key == &format!("layer {l} routed block") && detail.contains(fragment)));
        }
        let Ffn::Moe(moe) = &mut spec.layers[1].ffn else {
            panic!("routed");
        };
        moe.act = Act::ReluSqr;
        let error = layer(src, &spec, 1)
            .err()
            .expect("non-gated activation refused");
        assert!(matches!(error, ModelError::Metadata { ref detail, .. }
            if detail.contains("this host runs SwiGLU and the gate-first clamp")));
        std::fs::remove_file(path).unwrap();
    }

    /// A routed layer builds whatever clamp its description states: the gate-first
    /// limit is the layer's own (`set_act`), not another layer's or a fixed one.
    /// Mutant: the `set_act` call dropped — the clamped layer runs the plain combine.
    #[test]
    fn a_layer_builds_its_own_clamp_order() {
        // The stacks in Q8_0 need rows of 32 values, so this header widens the
        // tiny one's model width to 32 and its expert widths to 32 and 64.
        let kv: Vec<_> = keys()
            .into_iter()
            .map(|(k, v)| {
                let widened = match k {
                    "embedding_length" => Some(V::U32(32)),
                    "expert_feed_forward_length" => Some(V::U32(32)),
                    "expert_shared_feed_forward_length" => Some(V::U32(64)),
                    _ => None,
                };
                (k, widened.unwrap_or(v))
            })
            .collect();
        let widen = |name: &str, dims: &[u64]| -> Vec<u64> {
            let e = 32_u64;
            match name.rsplit('.').nth(1).unwrap_or("") {
                // The model width is 32; the expert widths 32 and 64; every
                // other dim (heads, the expert count, the dense width) as it was.
                "attn_norm" | "ffn_norm" | "output_norm" => vec![e],
                "token_embd" | "output" => vec![e, dims[1]],
                "attn_q" | "ffn_gate" | "ffn_up" => vec![e, dims[1]],
                "attn_k" | "attn_v" | "ffn_gate_inp" => vec![e, dims[1]],
                "attn_output" | "ffn_down" => vec![dims[0], e],
                "ffn_gate_shexp" | "ffn_up_shexp" => vec![e, 64],
                "ffn_down_shexp" => vec![64, e],
                "ffn_gate_exps" | "ffn_up_exps" => vec![e, 32, dims[2]],
                "ffn_down_exps" => vec![32, e, dims[2]],
                _ => dims.to_vec(),
            }
        };
        let typed: Vec<_> = tensors()
            .into_iter()
            .map(|(name, dims)| {
                let dims = widen(&name, &dims);
                let exps = name.ends_with("ffn_gate_exps.weight")
                    || name.ends_with("ffn_up_exps.weight")
                    || name.ends_with("ffn_down_exps.weight");
                (name, dims, if exps { Ty::Q8_0 } else { Ty::F32 })
            })
            .collect();
        let path = header_typed(
            "flint-host-clamp",
            "flint",
            &kv,
            &[("tokenizer.ggml.pre", V::Str("flint"))],
            &typed,
        );
        let split = gguf::Split::open(&path).unwrap();
        let mut spec = super::super::spec::read(&split).unwrap().spec;
        let Ffn::Moe(moe) = &mut spec.layers[1].ffn else {
            panic!("routed");
        };
        moe.act = Act::SwiGluGateFirst { limit: 7.0 };
        let built = layer(R8Source::rows(&split), &spec, 1).unwrap();
        assert_eq!(
            built.form(),
            ExpertForm::GatedGateFirst { limit: 7.0 },
            "the clamped layer carries its own gate-first limit"
        );
        std::fs::remove_file(path).unwrap();
    }
}
