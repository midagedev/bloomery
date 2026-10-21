//! An M3 file planned onto a machine: the common hybrid owner's plan ([`place_hybrid`]) over this
//! family's reading. What each layer holds is read off the description: every layer's K and V
//! planes, and for a sparse layer ([`models::Selector::BlockMax`]) an f16 index key of the
//! indexer's width for every position ([`KvLayout`]). The card experts' formats are the common
//! leg's, and a machine that hangs an expert tier card is refused by name ([`PlaceError::Tier`])
//! until the body serves one. The family's own is the reading ([`PlanInputs::describe`]) and the
//! engine's name in the refusals.

use gguf::Split;

use super::hparams::Hparams;
use super::{roles, spec};
use crate::arch::chat_of;
use crate::arch::place_hybrid;
use crate::placement::PlacementError;

pub use place_hybrid::{KvLayout, PlaceError, card_routed};

/// What a plan of an M3 file is made from, read from its headers.
pub type PlanInputs = place_hybrid::PlanInputs<Hparams>;

/// The engine the plan's refusals name.
const ENGINE: &str = "MiniMax-M3";

impl PlanInputs {
    /// [`PlanInputs::describe`], then the file refused with every feature the engine does not run
    /// ([`PlacementError::Unimplemented`]). The reader refuses a next-token block by its tensors.
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        PlanInputs::describe(split)?.runnable()
    }

    /// The file as it is, for a gate that plans or describes it whatever the coverage list holds,
    /// and the host's room and the arena's given bytes. The prompt batch's bytes come out of the
    /// plan's scratch term, so the plan holds no card bytes beside its terms
    /// ([`PlanInputs::prompt_reserve`] is 0).
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let model = roles::classify(split, &hp)?;
        // No parser is declared: see `spec`.
        let spec = spec::spec_of(&hp, chat_of(split, None, None)?)?;
        Ok(PlanInputs::new(ENGINE, hp, model, spec)?.with_prompt_reserve(0))
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use gguf::Split;

    use super::super::hparams::Kind;
    use super::super::hparams::tests::{IDX_D, LAYERS, keys, tensors};
    use super::{PlaceError, PlanInputs};
    use crate::arch::synthetic::{V, header_shaped};
    use crate::placement::{PlacementError, PlanLevers, workstation};

    /// The small header's global keys: the pre-tokenizer and a template the description reads.
    fn global() -> [(&'static str, V); 2] {
        [
            ("tokenizer.ggml.pre", V::Str("minimax-m2")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ]
    }

    /// A header, as the small file the `hparams` tests describe; each test names its own file (the
    /// tests run in threads of one process).
    fn written(name: &str) -> std::path::PathBuf {
        header_shaped(name, "minimax-m3", &keys(), &global(), &tensors())
    }

    /// The small header's file, described.
    fn described(name: &str) -> PlanInputs {
        let path = written(name);
        let split = Split::open(&path).expect("the synthetic header opens");
        let inputs = PlanInputs::describe(&split).expect("the synthetic header describes");
        let _ = std::fs::remove_file(&path);
        inputs
    }

    /// A header reads into the plan's inputs; the file is refused, every missing feature listed,
    /// until a program runs it. A file the reader refuses is that refusal, not a coverage one.
    #[test]
    fn the_inputs_describe_a_file_and_the_coverage_refuses_it() {
        let path = written("mm3-place");
        let split = Split::open(&path).expect("the synthetic header opens");
        let described = PlanInputs::describe(&split).expect("the header describes");
        let refused = PlanInputs::read(&split);
        let _ = std::fs::remove_file(&path);
        assert_eq!(described.model.layers, 5);
        assert!(described.hp.kind(2) == Kind::Sparse && described.hp.kind(1) == Kind::Dense);
        assert_eq!(described.prompt_reserve, 0);
        match refused {
            Err(PlacementError::Unimplemented(items)) => {
                assert!(
                    items.iter().any(|u| u.feature.contains("swigluoai")),
                    "{items:?}"
                );
            }
            other => panic!("the coverage refusal: {other:?}"),
        }
        let bad = header_shaped(
            "mm3-place-bad",
            "minimax-m3",
            &[("block_count", V::U32(5))],
            &global(),
            &tensors(),
        );
        let split = Split::open(&bad).expect("the synthetic header opens");
        let err = PlanInputs::read(&split).expect_err("a header without keys");
        let _ = std::fs::remove_file(&bad);
        assert!(matches!(err, PlacementError::Metadata { .. }), "{err:?}");
    }

    /// The card's kv class is the layers' stores [derived from the header's widths, apart from the
    /// reader]: 2 key heads × (16 + 16) × 2 B = 128 B of K and V a position, and the 3 sparse
    /// layers (2, 3 and 4) 16 f16 index-key values = 32 B more; at 64 positions the 2 dense layers
    /// are 8,192 B each and the 3 sparse 10,240 B each, 47,104 B, a slot each time. The stage keeps
    /// one table of the 8 rotated values in f32 (32 B) a position, 2,048 B at 64, whatever the
    /// slots. A machine that hangs an expert tier card is refused by name, and a context of no
    /// position first. Mutant: the index keys left out of the layout (the card holds 40,960 B), or
    /// the stage rows counted a slot.
    #[test]
    fn the_cards_kv_class_is_the_layers_stores_and_a_tier_card_is_refused() {
        let inputs = described("mm3-place-kv");
        assert_eq!(inputs.kv.bytes(0..LAYERS, 64), 47_104);
        let machine = workstation::plan_gate(LAYERS);
        let levers = PlanLevers::default();
        let slots = |n| NonZeroUsize::new(n).expect("a slot");
        for n in [1, 3] {
            let plan = inputs
                .plan_with_slots(&machine, 64, &levers, slots(n))
                .expect("the small header plans");
            let want = n as u64 * 47_104;
            assert_eq!(plan.cards[0].kv_bytes, want, "{n} slots");
            assert_eq!(want, n as u64 * inputs.kv.bytes(0..LAYERS, 64), "{n} slots");
            // The rope table and the index selector's sink row, a position each.
            let rows = 8 * 4 + IDX_D * 2;
            assert_eq!(plan.cards[0].stage_rows_bytes, rows * 64, "{n} slots");
        }
        let mut tiered = workstation::plan_gate(LAYERS);
        let mut tier = tiered.cards[0].clone();
        tier.layers = 0..0;
        tier.head = false;
        tier.token_embedding = false;
        tiered.tiers.push(tier);
        match inputs.plan(&tiered, 64, &levers) {
            Err(e @ PlaceError::Tier { tier: 0, .. }) => assert_eq!(
                e.to_string(),
                "the machine has the expert tier card 3090 (tier 0); the program this plan is \
                 for hangs no expert tier"
            ),
            other => panic!("{other:?}, not the tier refusal"),
        }
        assert!(matches!(
            inputs.plan(&tiered, 0, &levers),
            Err(PlaceError::CtxZero)
        ));
    }
}
