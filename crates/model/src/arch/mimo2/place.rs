//! A mimo2 file planned onto a machine: the common hybrid owner's plan
//! ([`place_hybrid`]) over this family's reading, as `glm5next::place` is for
//! GLM. The hyperparameters, every tensor's role and the typed description are
//! read here ([`PlanInputs::read`], which refuses a file whose next-token
//! layers carry tensors — the draft that reads them is a later round's — and
//! one with a feature no program runs, every one listed); what each layer
//! holds, the expert rule and the refusals are the owner's.

use gguf::Split;

use super::hparams::Hparams;
use super::{roles, spec};
use crate::arch::chat_of;
use crate::arch::place_hybrid;
use crate::placement::PlacementError;

// The ring rule's owner is `runtime::stores`; these names stay on this path for
// `gpu-mimo2` and the gates, which import them from here.
pub use place_hybrid::{KvLayout, PlaceError, card_routed};
pub use runtime::stores::{Rows, cut_depth_keeps};

/// What a plan of a mimo2 file is made from, read from its headers.
pub type PlanInputs = place_hybrid::PlanInputs<Hparams>;

/// The engine the plan's refusals name.
const ENGINE: &str = "MiMo";

impl PlanInputs {
    /// `split`'s hyperparameters, then its tensors' roles, then its
    /// description, the first that fails being the error; then a file whose
    /// next-token layers carry tensors is refused by the next-token count's
    /// name (the draft that reads them is a later round's; this plan loads
    /// the trunk without them), and one with a feature the engine does not
    /// run ([`PlacementError::Unimplemented`], every one listed).
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        let inputs = PlanInputs::describe(split)?;
        if inputs.hp.n_layer > inputs.hp.n_trunk {
            return Err(PlacementError::Metadata {
                key: "nextn_predict_layers".to_string(),
                detail: format!(
                    "is {}; the draft that reads the next-token layers is a later round's, and \
                     this plan loads the trunk without them",
                    inputs.hp.n_layer - inputs.hp.n_trunk
                ),
            });
        }
        inputs.runnable()
    }

    /// [`PlanInputs::read`] without the refusals: the file as it is, for a
    /// gate that plans or describes a file whatever the coverage list holds,
    /// and the host's room and the arena's given bytes, a room that cannot be
    /// read refused by name.
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let model = roles::classify(split, &hp)?;
        let chat = chat_of(split, spec::TOOLS, Some(models::ReasoningFormat::THINK))?;
        let spec = spec::spec_of(&hp, &model, chat)?;
        PlanInputs::new(ENGINE, hp, model, spec)
    }
}

#[cfg(test)]
mod tests {
    use gguf::Split;

    use super::super::hparams::tests::{keys, tensors};
    use super::{PlaceError, PlanInputs};
    use crate::arch::synthetic::{V, header_shaped};
    use crate::placement::{KvBytes, PlanLevers, workstation};

    /// A file whose next-token layers carry tensors is refused at
    /// [`PlanInputs::read`] by the next-token count's name — the draft that
    /// reads them is a later round's, and this plan loads the trunk without
    /// them — before the coverage list; the same file without its next-token
    /// layer is refused by that list alone (the synthetic head width is one no
    /// row of the mimo2 body runs), so the count is the named refusal.
    #[test]
    fn next_token_layers_are_refused_at_read() {
        let err = read_inputs("mimo2-place-nextn", &keys(), &tensors())
            .expect_err("a file with a next-token layer is refused");
        assert!(
            err.contains("metadata nextn_predict_layers: is 1")
                && err.contains("the draft that reads the next-token layers is a later round's"),
            "{err}"
        );
        let tensors: Vec<(String, Vec<u64>)> = tensors()
            .into_iter()
            .filter(|(n, _)| !n.starts_with("blk.4."))
            .collect();
        let err = read_inputs("mimo2-place-trunk", &trunk_keys(), &tensors)
            .expect_err("a trunk-only file is refused by the coverage list");
        assert!(
            err.contains("feature(s) of this file are not implemented"),
            "{err}"
        );
    }

    /// A machine that hangs an expert tier card is refused by name, before
    /// any plan is made: the program's card experts have no tier to join,
    /// and a context of no position is refused first.
    #[test]
    fn a_tier_card_is_refused_by_name() {
        let path = header_shaped(
            "mimo2-place-tier",
            "mimo2",
            &keys(),
            &[
                ("tokenizer.ggml.pre", V::Str("qwen2")),
                ("tokenizer.chat_template", V::Str("{{ messages }}")),
            ],
            &tensors(),
        );
        let split = Split::open(&path).expect("the synthetic header opens");
        let inputs = PlanInputs::describe(&split).expect("the synthetic header describes");
        let _ = std::fs::remove_file(&path);
        let mut machine = workstation::plan_gate(inputs.hp.n_trunk);
        let mut tier = machine.cards[0].clone();
        tier.layers = 0..0;
        tier.head = false;
        tier.token_embedding = false;
        machine.tiers.push(tier);
        let levers = PlanLevers::default();
        match inputs.plan(&machine, 4, &levers) {
            Err(e @ PlaceError::Tier { tier: 0, .. }) => assert_eq!(
                e.to_string(),
                "the machine has the expert tier card 3090 (tier 0); the program this plan is \
                 for hangs no expert tier"
            ),
            other => panic!("{other:?}, not the tier refusal"),
        }
        assert!(matches!(
            inputs.plan(&machine, 0, &levers),
            Err(PlaceError::CtxZero)
        ));
    }

    /// The small header of the `hparams` tests' keys with its next-token
    /// layer taken out: four trunk blocks, the per-layer arrays four long.
    fn trunk_keys() -> Vec<(&'static str, V)> {
        keys()
            .into_iter()
            .map(|(k, v)| match k {
                "block_count" => ("block_count", V::U32(4)),
                "nextn_predict_layers" => ("nextn_predict_layers", V::U32(0)),
                "attention.head_count_kv" => ("attention.head_count_kv", V::I32s(vec![2, 1, 1, 2])),
                "attention.sliding_window_pattern" => (
                    "attention.sliding_window_pattern",
                    V::I32s(vec![0, 1, 1, 0]),
                ),
                _ => (k, v),
            })
            .collect()
    }

    /// The stage's rope rows: one table a distinct rope base the trunk turns
    /// by, `rope.dimension_count` f32 values a position, as the body copies
    /// them to its card. The small header's full and window bases differ (two
    /// tables of 4 values: 32 B a position); one base for both is one table.
    /// The plan counts them on the card that runs layer 0, at the plan's
    /// positions. FAIL-first: a layout that counts no row leaves the card's
    /// `stage_rows_bytes` at 0 and the load's scratch check refuses a context
    /// whose rope tables pass the fixed scratch term.
    #[test]
    fn the_stage_rope_rows_are_a_table_a_base() {
        let describe = |tag: &str, kv: &[(&'static str, V)]| {
            let global = [
                ("tokenizer.ggml.pre", V::Str("qwen2")),
                ("tokenizer.chat_template", V::Str("{{ messages }}")),
            ];
            let path = header_shaped(tag, "mimo2", kv, &global, &tensors());
            let split = Split::open(&path).expect("the synthetic header opens");
            let inputs = PlanInputs::describe(&split).expect("the synthetic header describes");
            let _ = std::fs::remove_file(&path);
            inputs
        };
        let two = describe("mimo2-rope-two", &keys());
        assert_eq!(two.kv.stage_position_bytes(10), 2 * 4 * 4 * 10);
        let one_base: Vec<_> = keys()
            .into_iter()
            .map(|(k, v)| match k {
                "rope.freq_base_swa" => (k, V::F32(1e7)),
                _ => (k, v),
            })
            .collect();
        let one = describe("mimo2-rope-one", &one_base);
        assert_eq!(one.kv.stage_position_bytes(10), 4 * 4 * 10);
        let machine = workstation::plan_gate(two.hp.n_trunk);
        let plan = two
            .plan(&machine, 1024, &PlanLevers::default())
            .expect("the small header plans at 1,024");
        assert_eq!(plan.cards[0].stage_rows_bytes, 32 * 1024);
        assert_eq!(
            two.seq_terms().slots_of(3).stage_position_bytes(1024),
            32 * 1024
        );
    }

    /// [`PlanInputs::read`] of a synthetic header, or its error's text.
    fn read_inputs(
        tag: &str,
        kv: &[(&'static str, V)],
        tensors: &[(String, Vec<u64>)],
    ) -> Result<PlanInputs, String> {
        let global = [
            ("tokenizer.ggml.pre", V::Str("qwen2")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let path = header_shaped(tag, "mimo2", kv, &global, tensors);
        let split = Split::open(&path).expect("the synthetic header opens");
        let read = PlanInputs::read(&split).map_err(|e| e.to_string());
        let _ = std::fs::remove_file(&path);
        read
    }
}
