//! A Kimi K3 file planned onto a machine: the common hybrid owner's plan ([`place_hybrid`]) over
//! this family's reading. What each layer holds is read off the description (a KDA layer's state
//! and conv ring, a latent layer's plane of cached rows), the routed experts' bytes are the
//! file's by their stacks' types, and a machine that hangs an expert tier card is refused by name
//! ([`PlaceError::Tier`]) until a body serves one. No body runs the architecture yet, so
//! [`PlanInputs::read`] refuses the file with every feature the coverage check lists
//! ([`PlacementError::Unimplemented`]) and [`PlanInputs::plan`] with the same list: a plan of K3
//! is never made as if a program ran it.

use gguf::Split;

use super::hparams::Hparams;
use super::spec;
use crate::arch::place_hybrid;
use crate::placement::PlacementError;

pub use place_hybrid::{KvLayout, PlaceError};

/// What a plan of a Kimi K3 file is made from, read from its headers.
pub type PlanInputs = place_hybrid::PlanInputs<Hparams>;

/// The engine the plan's refusals name.
const ENGINE: &str = "Kimi K3";

impl PlanInputs {
    /// [`PlanInputs::describe`], then the file refused with every feature the engine does not run
    /// ([`PlacementError::Unimplemented`]).
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        PlanInputs::describe(split)?.runnable()
    }

    /// The file as it is, for a gate that describes it whatever the coverage list holds, and the
    /// host's room and the arena's given bytes. The roles and the description are the registry's
    /// reader's ([`spec::read`]), so the plan and the description of a file are one reading.
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let read = spec::read(split)?;
        PlanInputs::new(ENGINE, hp, read.tensors, read.spec)
    }
}

#[cfg(test)]
mod tests {
    use gguf::Split;

    use super::super::hparams::tests::{keys, tensors, typed};
    use super::{PlaceError, PlanInputs};
    use crate::arch::synthetic::{Ty, V, header_typed};
    use crate::placement::workstation::{self, A6000, RTX_3090};
    use crate::placement::{KvBytes, PlacementError, PlanLevers};

    /// The small header's file with `over` typed as the released set's other stacks, opened; each
    /// test names its own file (the tests run in threads of one process).
    fn opened(name: &str, over: &[(&str, Ty)]) -> Split {
        let global = [
            ("tokenizer.ggml.pre", V::Str("kimi-k2")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let path = header_typed(name, "kimi-k3", &keys(), &global, &typed(&tensors(), over));
        let split = Split::open(&path).expect("the synthetic header opens");
        let _ = std::fs::remove_file(&path);
        split
    }

    /// The small header's file, described.
    fn described(name: &str) -> PlanInputs {
        PlanInputs::describe(&opened(name, &[])).expect("the synthetic header describes")
    }

    /// A card's kv class is the layers' stores [derived from the header's widths, apart from the
    /// reader]: a KDA layer 2 heads of 128 x 128 f32 (131,072 B) + 11 ring rows (3 + 8 of a
    /// pass) of the q, k and v channels, 3 x 2 x 128 f32 (33,792 B) + a lane's u32 stamp =
    /// 164,868 B whatever the context; a latent layer a row of 512 + 64 f16 values (1,152 B) a
    /// position, 64 positions 73,728 B. At 64 positions the 3 KDA and 1 latent layers hold
    /// 3 x 164,868 + 73,728 = 568,332 B, a layer past the model none, and nothing rotates, so no
    /// rope row. Mutants (`.mutants/kimik3place-*.patch`): the state's bytes halved in
    /// `runtime::stores::recurrent_bytes`, the unrotated values left out of a latent row.
    #[test]
    fn the_layers_stores_are_the_headers_widths() {
        let inputs = described("kimi-k3-place-kv");
        let kv: &super::KvLayout = &inputs.kv;
        for l in [0, 1, 3] {
            assert_eq!(kv.layer_bytes(l, 1), 164_868, "layer {l}");
            assert_eq!(kv.layer_bytes(l, 1 << 20), 164_868, "layer {l}");
        }
        assert_eq!(kv.layer_bytes(2, 1), 1_152);
        assert_eq!(kv.layer_bytes(2, 64), 73_728);
        assert_eq!(kv.layer_bytes(4, 64), 0);
        assert_eq!(kv.bytes(0..4, 64), 3 * 164_868 + 73_728);
        assert_eq!(kv.stage_position_bytes(64), 0, "no layer rotates");
    }

    /// A routed stack is the file's bytes by its type [derived from the header's widths, apart
    /// from the reader]: an expert of the mini's 256 x 256 stack is 12,800 B in IQ1_S and 16,896 B
    /// in IQ2_XXS, so an expert (three stacks) is 38,400 B all IQ1_S, 46,592 B with the gate and
    /// up IQ2_XXS and the down IQ1_S, and 42,496 B with the down IQ2_XXS.
    #[test]
    fn an_expert_is_its_stacks_bytes_by_type() {
        let over = [
            ("blk.2.ffn_gate_exps.weight", Ty::Iq2Xxs),
            ("blk.2.ffn_up_exps.weight", Ty::Iq2Xxs),
            ("blk.3.ffn_down_exps.weight", Ty::Iq2Xxs),
        ];
        let inputs = PlanInputs::describe(&opened("kimi-k3-place-experts", &over))
            .expect("the synthetic header describes");
        let expert = |layer: usize| -> u64 {
            let experts = ["gate", "up", "down"].map(|stack| {
                let name = format!("blk.{layer}.ffn_{stack}_exps.weight");
                let t = inputs
                    .model
                    .tensors
                    .iter()
                    .find(|t| t.name == name)
                    .unwrap_or_else(|| panic!("{name} is in the file"));
                assert_eq!(t.dims[2], 4, "{name}: the stack's experts");
                t.file_bytes / 4
            });
            experts.iter().sum()
        };
        assert_eq!(expert(1), 38_400);
        assert_eq!(expert(2), 46_592);
        assert_eq!(expert(3), 42_496);
    }

    /// The file is described and the engine refuses to run it: [`PlanInputs::read`] and
    /// [`PlanInputs::plan`] each name every feature the coverage check lists, the same list; a
    /// context of no position is refused by its own name. The refusal lists go red when a body's
    /// program covers the architecture: that change re-pins them.
    #[test]
    fn the_described_file_is_refused_until_a_body_runs_it() {
        let inputs = described("kimi-k3-place-refused");
        let want = inputs.unimplemented();
        assert!(!want.is_empty(), "no body runs Kimi K3 yet");
        match PlanInputs::read(&opened("kimi-k3-place-read", &[])) {
            Err(PlacementError::Unimplemented(items)) => assert_eq!(items, want),
            other => panic!("{other:?}, not the coverage refusal"),
        }
        let machine = workstation::plan_gate(4);
        match inputs.plan(&machine, 64, &PlanLevers::default()) {
            Err(PlaceError::Placement(PlacementError::Unimplemented(items))) => {
                assert_eq!(items, want);
            }
            other => panic!("{other:?}, not the coverage refusal"),
        }
        assert!(matches!(
            inputs.plan(&machine, 0, &PlanLevers::default()),
            Err(PlaceError::CtxZero)
        ));
    }

    /// A machine that hangs an expert tier card is refused by the tier's name, before the
    /// coverage list: no body serves a tier.
    #[test]
    fn a_tier_card_is_refused_by_name() {
        let inputs = described("kimi-k3-place-tier");
        let machine = workstation::plan_tiers(
            4,
            A6000,
            &[RTX_3090],
            None,
            workstation::tier_batch_bytes(512, 256, 2, 512),
        )
        .expect("the tier machine plans");
        match inputs.plan(&machine, 64, &PlanLevers::default()) {
            Err(PlaceError::Tier { card, tier }) => assert_eq!((card.as_str(), tier), ("3090", 0)),
            other => panic!("{other:?}, not the tier refusal"),
        }
    }

    /// The plan's description is the registry's: one reading of the file, whose chat surface is
    /// the Kimi K2 pre-tokenizer and the file's template, with no tool-call or reasoning parser
    /// named.
    #[test]
    fn the_plans_description_is_the_registrys() {
        let split = opened("kimi-k3-place-chat", &[]);
        let inputs = PlanInputs::describe(&split).expect("describes");
        let read = crate::arch::spec(&split).expect("the registry reads");
        assert_eq!(inputs.spec, read.spec);
        let chat = &inputs.spec.chat;
        assert_eq!(chat.pre, "kimi-k2");
        assert_eq!(chat.template.as_deref(), Some("{{ messages }}"));
        assert!(chat.tools.is_none() && chat.reasoning.is_none());
    }
}
