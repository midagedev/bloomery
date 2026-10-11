//! The qwen3moe chain's routed stack table for the residency machine
//! ([`crate::host::swap`]), one owner for every body of the family: the common
//! table ([`PlanStacks`]) over the family's facts, each layer's stacks checked
//! against the types the body's card leg runs them in. A body names its leg as
//! data ([`CardLeg`]: what its refusals call it, and the types its gate·up and
//! down launches read) and builds its table with [`stacks`]: `swap35` is
//! Qwen3.6's, `swap38` Qwen3.8's.
//!
//! **The check.** A layer the body's plan gives card experts, whose three
//! stacks are all of types the leg runs (the place rule's card stacks), holds a gate and an up of one type its
//! gate·up launches read and a down of a type its down launches read; a layer
//! with a stack of another type stays on the host and is not checked, and so
//! is a layer the plan gives no card expert: the machine keeps each layer's
//! card count, so none of its stacks ever reaches the leg. The
//! place rule reads a stack's type alone, so a type the leg runs in one role
//! only (a Q6_K gate, a Q4_K down on Qwen3.8's leg) puts its layer on the
//! card, and the check refuses it by name here, at load, instead of the leg
//! at its first launch.

use crate::GpuError;
use crate::host::swap_source::{FileStacks, PlanStacks, StackFacts};
use gguf::quant::GgmlType;
use model::arch::names::routed_stacks;
use model::placement::ModelTensors;

/// A body's card leg as data: the types its launches read a routed stack in,
/// by role.
pub(super) trait CardLeg {
    /// What the table's refusals name.
    const WHAT: &'static str;

    /// Whether the leg's gate·up launches read a gate or an up stack of `ty`.
    fn gate_up(ty: GgmlType) -> bool;

    /// Whether the leg's down launches read a down stack of `ty`.
    fn down(ty: GgmlType) -> bool;
}

/// The file types `pred` holds, in the file format's id order: the lists a
/// refusal names.
fn listed(pred: fn(GgmlType) -> bool) -> Vec<GgmlType> {
    (0..64)
        .map(GgmlType::from_u32)
        .filter(|&ty| !matches!(ty, GgmlType::Unknown(_)) && pred(ty))
        .collect()
}

/// Layer `l`'s card-leg contract under the leg `L` (module doc): a layer whose stacks are all card types holds a gate
/// and an up of one gate·up type and a down type the down launches read;
/// refused by name otherwise.
fn check<L: CardLeg>(l: usize, [gate, up, down]: [GgmlType; 3]) -> Result<(), GpuError> {
    let carded = [gate, up, down]
        .iter()
        .all(|&ty| L::gate_up(ty) || L::down(ty));
    if carded && (gate != up || !L::gate_up(gate) || !L::down(down)) {
        return Err(GpuError::shape(
            L::WHAT,
            format!(
                "layer {l}'s gate {gate}, up {up} and down {down}: the card leg runs a \
                 gate and an up of one type, {:?}, and a down of {:?}",
                listed(L::gate_up),
                listed(L::down)
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

    /// A leg runs a stack of a type in some role exactly where the family's
    /// place rule loads it on the card: Qwen3.6's the place rule of
    /// `model::arch::qwen3moe::place`, Qwen3.8's that of
    /// `model::arch::qwen35moe::place`. Mutant: a type dropped from a leg's
    /// role (or added to it) — the sweep names it.
    #[test]
    fn a_leg_runs_the_types_its_place_rule_loads() {
        for ty in (0..64).map(GgmlType::from_u32) {
            assert_eq!(
                Leg35::gate_up(ty) || Leg35::down(ty),
                model::arch::qwen3moe::place::card_routed(ty).is_some(),
                "Qwen3.6, {ty}"
            );
            assert_eq!(
                Leg38::gate_up(ty) || Leg38::down(ty),
                model::arch::qwen35moe::place::card_routed(ty).is_some(),
                "Qwen3.8, {ty}"
            );
        }
    }

    /// Qwen3.6's leg: a Q4_K or Q5_K gate and up of one type with a Q4_K,
    /// Q5_K or Q6_K down is a card layer; a Q6_K gate and up (a type its
    /// down runs, not its gate·up), a gate and up of two types and a Q6_K up
    /// beside a Q4_K gate are refused by name; a layer with a stack of a type
    /// the leg does not run (a Q5_1 down, an IQ4_XS gate) stays on the host
    /// unchecked. Mutant: Qwen3.8's down list for Qwen3.6's — the Q4_K and
    /// Q6_K downs are refused (the first arms) or let through (the Q5_1 arm
    /// turns card).
    #[test]
    fn qwen36s_leg_checks_a_card_layer_against_its_launches() {
        use GgmlType::{IQ4_XS, Q4_K, Q5_1, Q5_K, Q6_K};
        for ok in [
            [Q4_K, Q4_K, Q4_K],
            [Q4_K, Q4_K, Q6_K],
            [Q5_K, Q5_K, Q5_K],
            [Q5_K, Q5_K, Q6_K],
            [Q4_K, Q4_K, Q5_1],
            [IQ4_XS, IQ4_XS, Q4_K],
        ] {
            assert_eq!(layer::<Leg35>(ok), Ok(()), "{ok:?}");
        }
        for bad in [
            [Q6_K, Q6_K, Q4_K],
            [Q4_K, Q5_K, Q4_K],
            [Q4_K, Q6_K, Q4_K],
            [Q6_K, Q6_K, Q6_K],
        ] {
            let text = layer::<Leg35>(bad).expect_err(&format!("{bad:?} is refused"));
            assert!(
                text.contains("layer 0's gate")
                    && text.ends_with("a gate and an up of one type, [Q4_K, Q5_K], and a down of [Q4_K, Q5_K, Q6_K]"),
                "{text}"
            );
        }
    }

    /// Qwen3.8's leg, as its table held it: a Q4_K, Q5_K, IQ3_XXS or IQ4_XS
    /// gate and up of one type with a Q5_1, Q8_0 or IQ4_NL down is a card
    /// layer, and the refusal's text lists the types as the table did; a Q4_K
    /// down is refused (a down type its gate·up reads), a Q6_K one is not a
    /// card type and stays on the host. Mutant: Qwen3.6's down for Qwen3.8's
    /// — the Q5_1 and IQ4_NL arms turn host, the Q4_K arm passes.
    #[test]
    fn qwen38s_leg_keeps_its_table() {
        use GgmlType::{IQ3_XXS, IQ4_NL, IQ4_XS, Q4_K, Q5_1, Q5_K, Q6_K, Q8_0};
        for ok in [
            [Q4_K, Q4_K, Q5_1],
            [Q5_K, Q5_K, Q8_0],
            [IQ3_XXS, IQ3_XXS, IQ4_NL],
            [IQ4_XS, IQ4_XS, Q8_0],
            [Q4_K, Q4_K, Q6_K],
        ] {
            assert_eq!(layer::<Leg38>(ok), Ok(()), "{ok:?}");
        }
        for bad in [[Q4_K, Q4_K, Q4_K], [Q4_K, Q5_K, Q5_1], [Q5_1, Q5_1, Q5_1]] {
            let text = layer::<Leg38>(bad).expect_err(&format!("{bad:?} is refused"));
            assert_eq!(
                text,
                format!(
                    "layer 0's gate {}, up {} and down {}: the card leg runs a gate and an up \
                     of one type, [Q4_K, Q5_K, IQ3_XXS, IQ4_XS], and a down of [Q5_1, Q8_0, \
                     IQ4_NL]",
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
        use GgmlType::{Q4_K, Q6_K};
        let three = |l: usize| {
            if l == 2 {
                [Q6_K, Q6_K, Q4_K]
            } else {
                [Q4_K, Q4_K, Q6_K]
            }
        };
        match stacks::<Leg35>(&model(4, three), |_| Ok(true)) {
            Err(GpuError::Shape { detail, .. }) => {
                assert!(detail.starts_with("layer 2's gate q6_K"), "{detail}");
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
    /// leg never runs its stacks, so a gate and an up of two card types (or a
    /// Q6_K gate and up on Qwen3.6's leg) is refused only on a carded layer.
    /// Mutant: the check run on every layer whatever `carded` says — the
    /// uncarded arms are refused.
    #[test]
    fn a_layer_with_no_card_expert_is_not_checked() {
        use GgmlType::{Q4_K, Q5_K, Q6_K};
        for bad in [[Q4_K, Q5_K, Q4_K], [Q6_K, Q6_K, Q4_K]] {
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
