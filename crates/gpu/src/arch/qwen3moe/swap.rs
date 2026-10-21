//! The qwen3moe chain's routed stack table for the residency machine
//! ([`crate::host::swap`]), one owner for every body of the family: the common
//! table ([`PlanStacks`]) over the family's facts, each layer's stacks checked
//! against the types the body's card leg runs them in. A body names its leg
//! ([`CardLeg`]: what its refusals call it, and its program, whose leg arms
//! are `placement::accepts`'s) and builds its table with [`stacks`]: `swap35`
//! is Qwen3.6's, `swap38` Qwen3.8's.
//!
//! **The check.** A layer the body's plan gives card experts holds a gate and
//! an up of one type its gate·up launches read and a down of a type its down
//! launches read ([`accepts::leg_runs`], the arms the plan's rule reads too, so
//! a plan the rule passes holds no layer this refuses); a layer the plan gives
//! no card expert is not checked: the machine keeps each layer's card count,
//! so none of its stacks ever reaches the leg.

use crate::GpuError;
use crate::host::swap_source::{FileStacks, PlanStacks, StackFacts};
use gguf::quant::GgmlType;
use model::arch::names::routed_stacks;
use model::placement::ModelTensors;
use model::placement::accepts::{self, Program};

/// A body's card leg: what its refusals call it, and the program whose leg
/// arms it runs ([`accepts::leg_arms`]).
pub(super) trait CardLeg {
    /// What the table's refusals name.
    const WHAT: &'static str;

    /// The program the leg is.
    const PROGRAM: Program;
}

/// The file types `pred` holds, in the file format's id order: the lists a
/// refusal names.
fn listed(pred: fn(GgmlType) -> bool) -> Vec<GgmlType> {
    (0..64)
        .map(GgmlType::from_u32)
        .filter(|&ty| !matches!(ty, GgmlType::Unknown(_)) && pred(ty))
        .collect()
}

/// Layer `l`'s card-leg contract under the leg `L` (module doc): a gate and an
/// up of one gate·up type and a down type the down launches read; refused by
/// name otherwise, the arms' types listed.
fn check<L: CardLeg>(l: usize, [gate, up, down]: [GgmlType; 3]) -> Result<(), GpuError> {
    let arms = accepts::leg_arms(L::PROGRAM)
        .ok_or_else(|| GpuError::shape(L::WHAT, format!("{:?} runs no card leg", L::PROGRAM)))?;
    if accepts::leg_runs(L::PROGRAM, [gate, up, down]) != Some(true) {
        return Err(GpuError::shape(
            L::WHAT,
            format!(
                "layer {l}'s gate {gate}, up {up} and down {down}: the card leg runs a \
                 gate and an up of one type, {:?}, and a down of {:?}",
                listed(arms.gate_up),
                listed(arms.down)
            ),
        ));
    }
    Ok(())
}

/// The stacks of `model`'s layers, each layer's types read from its tensors:
/// every layer holds all three stacks, each layer `carded` names (one the
/// plan gives card experts) checked against the leg `L`, the whole card range
/// the map, every layer routed. Refused by name, under `L::WHAT`: a layer
/// without all three stacks, and a carded layer the leg refuses; a layer
/// `carded` refuses fails with its error. Load-time only.
pub(super) fn stacks<L: CardLeg>(
    model: &ModelTensors,
    carded: impl Fn(usize) -> Result<bool, GpuError>,
) -> Result<PlanStacks, GpuError> {
    let table = PlanStacks::of(
        model,
        StackFacts {
            what: L::WHAT,
            names: routed_stacks,
            check: None,
            empty: false,
            map_layers: None,
            unrouted: Vec::new(),
        },
    )?;
    for l in 0..model.layers {
        if let &[gate, up, down] = table.types(l)
            && carded(l)?
        {
            check::<L>(l, [gate, up, down])?;
        }
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::super::swap35::Leg35;
    use super::super::swap38::Leg38;
    use super::*;
    use model::placement::{ModelTensor, Role};

    /// A model whose layer `l` holds stacks of `types(l)` (gate, up, down),
    /// the eight experts of a stack no bigger than its bytes need.
    fn model(layers: usize, types: impl Fn(usize) -> [GgmlType; 3]) -> ModelTensors {
        let mut tensors = Vec::new();
        for l in 0..layers {
            for (name, ty) in routed_stacks(l).into_iter().zip(types(l)) {
                tensors.push(ModelTensor {
                    name,
                    shard: 0,
                    layer: Some(l),
                    role: Role::RoutedExperts,
                    ty,
                    dims: vec![256, 64, 8],
                    file_bytes: 1,
                    gathered_rows: None,
                });
            }
        }
        ModelTensors {
            tensors,
            layers,
            experts: 8,
            experts_used: 2,
            bf16_words: |_| false,
        }
    }

    /// What `L` makes of one layer of `types`: `Ok` or the refusal's text.
    fn layer<L: CardLeg>(types: [GgmlType; 3]) -> Result<(), String> {
        match stacks::<L>(&model(1, |_| types), |_| Ok(true)) {
            Ok(_) => Ok(()),
            Err(GpuError::Shape { what, detail }) => {
                assert_eq!(what, L::WHAT, "the refusal names the leg");
                Err(detail)
            }
            Err(e) => panic!("a shape refusal, not {e}"),
        }
    }

    /// Qwen3.6's leg: the dispatch's types (a Q4_K or Q5_K gate and up of one
    /// type with a Q4_K, Q5_K or Q6_K down) and the common leg's (a Q6_K, Q8_0,
    /// IQ3_XXS or IQ4_XS gate and up of one type with a down any of the
    /// common down arm's types) are card layers; a gate and up of two types,
    /// and a carded layer with a stack of a type no card launch runs (an
    /// IQ2_XS gate: no prompt GEMM, an MXFP4 down) are refused by name, the
    /// refusal listing both arms' types in file order. PIN(2026-10-22): the
    /// IQ4_XS and Q6_K gate and up and the Q5_1 down moved from refused to card
    /// layers when `placement::accepts` gave the body the common leg's arms.
    /// Mutant: the dispatch's arms (`accepts::qwen35_gate_up`/`qwen35_down`)
    /// for the common ones in `leg_arms` — the IQ4_XS, Q6_K, Q8_0 and IQ3_XXS
    /// layers are refused.
    #[test]
    fn qwen36s_leg_checks_a_card_layer_against_its_launches() {
        use GgmlType::{IQ2_XS, IQ3_XXS, IQ4_NL, IQ4_XS, MXFP4, Q4_K, Q5_1, Q5_K, Q6_K, Q8_0};
        for ok in [
            [Q4_K, Q4_K, Q4_K],
            [Q4_K, Q4_K, Q6_K],
            [Q5_K, Q5_K, Q5_K],
            [Q5_K, Q5_K, Q6_K],
            [Q4_K, Q4_K, Q5_1],
            [IQ4_XS, IQ4_XS, Q4_K],
            [Q8_0, Q8_0, Q8_0],
            [IQ3_XXS, IQ3_XXS, IQ4_NL],
            [Q6_K, Q6_K, Q4_K],
            [Q6_K, Q6_K, Q8_0],
        ] {
            assert_eq!(layer::<Leg35>(ok), Ok(()), "{ok:?}");
        }
        for bad in [
            [IQ2_XS, IQ2_XS, Q4_K],
            [Q4_K, Q4_K, MXFP4],
            [Q4_K, Q5_K, Q4_K],
            [Q4_K, Q6_K, Q4_K],
            [Q6_K, Q8_0, Q6_K],
        ] {
            let text = layer::<Leg35>(bad).expect_err(&format!("{bad:?} is refused"));
            assert!(
                text.contains("layer 0's gate")
                    && text.ends_with(
                        "a gate and an up of one type, [Q8_0, Q4_K, Q5_K, Q6_K, IQ3_XXS, IQ4_XS], \
                         and a down of [Q5_1, Q8_0, Q4_K, Q5_K, Q6_K, IQ3_XXS, IQ4_NL, IQ4_XS]"
                    ),
                "{text}"
            );
        }
    }

    /// Qwen3.8's leg, as its table held it: a Q4_K, Q5_K, IQ3_XXS or IQ4_XS
    /// gate and up of one type with a Q5_1, Q8_0 or IQ4_NL down is a card
    /// layer, and the refusal's text lists the types as the table did; a Q4_K
    /// down (a down type its gate·up reads) and a Q6_K one (a type no part
    /// reads) are refused. Mutant: Qwen3.6's down arm for Qwen3.8's — the
    /// Q5_1 and IQ4_NL arms are refused, the Q4_K one passes.
    #[test]
    fn qwen38s_leg_keeps_its_table() {
        use GgmlType::{IQ3_XXS, IQ4_NL, IQ4_XS, NVFP4, Q4_K, Q5_1, Q5_K, Q6_K, Q8_0};
        for ok in [
            [Q4_K, Q4_K, Q5_1],
            [Q5_K, Q5_K, Q8_0],
            [IQ3_XXS, IQ3_XXS, IQ4_NL],
            [IQ4_XS, IQ4_XS, Q8_0],
            [NVFP4, NVFP4, NVFP4],
        ] {
            assert_eq!(layer::<Leg38>(ok), Ok(()), "{ok:?}");
        }
        for bad in [
            [Q4_K, Q4_K, Q4_K],
            [Q4_K, Q4_K, Q6_K],
            [Q4_K, Q5_K, Q5_1],
            [Q5_1, Q5_1, Q5_1],
        ] {
            let text = layer::<Leg38>(bad).expect_err(&format!("{bad:?} is refused"));
            assert_eq!(
                text,
                format!(
                    "layer 0's gate {}, up {} and down {}: the card leg runs a gate and an up \
                     of one type, [Q4_K, Q5_K, IQ3_XXS, IQ4_XS, NVFP4], and a down of [Q5_1, \
                     Q8_0, IQ4_NL, NVFP4]",
                    bad[0], bad[1], bad[2]
                )
            );
        }
    }

    /// The refusal names the layer that breaks the contract, whatever layers
    /// come before it, and a layer missing a stack is refused by the common
    /// table's own name. Mutant: the check run on the first layer only — the
    /// layer 2 arm catches it.
    #[test]
    fn the_table_checks_every_layer() {
        use GgmlType::{IQ2_XS, Q4_K, Q6_K};
        let three = |l: usize| {
            if l == 2 {
                [IQ2_XS, IQ2_XS, Q4_K]
            } else {
                [Q4_K, Q4_K, Q6_K]
            }
        };
        match stacks::<Leg35>(&model(4, three), |_| Ok(true)) {
            Err(GpuError::Shape { detail, .. }) => {
                assert!(detail.starts_with("layer 2's gate iq2_xs"), "{detail}");
            }
            _ => panic!("layer 2 is refused by name"),
        }
        let mut short = model(2, |_| [Q4_K, Q4_K, Q6_K]);
        short.tensors.pop();
        assert!(
            matches!(
                stacks::<Leg35>(&short, |_| Ok(true)),
                Err(GpuError::Tensor { .. })
            ),
            "a layer without its down is a missing tensor"
        );
    }

    /// A layer the plan gives no card expert loads whatever its types: the
    /// leg never runs its stacks, so a gate and an up of two card types (or an
    /// IQ2_XS gate and up on Qwen3.6's leg) is refused only on a carded layer.
    /// Mutant: the check run on every layer whatever `carded` says — the
    /// uncarded arms are refused.
    #[test]
    fn a_layer_with_no_card_expert_is_not_checked() {
        use GgmlType::{IQ2_XS, Q4_K, Q5_K};
        for bad in [[Q4_K, Q5_K, Q4_K], [IQ2_XS, IQ2_XS, Q4_K]] {
            let two = model(2, |_| bad);
            assert!(
                stacks::<Leg35>(&two, |_| Ok(false)).is_ok(),
                "{bad:?} on layers with no card expert loads"
            );
            match stacks::<Leg35>(&two, |l| Ok(l == 1)) {
                Err(GpuError::Shape { detail, .. }) => {
                    assert!(detail.starts_with("layer 1's gate"), "{detail}");
                }
                _ => panic!("{bad:?} on carded layer 1 is refused by name"),
            }
        }
    }
}
