//! A `nemotron_h` file planned onto a machine: the family's half of the one
//! path from the file's headers to a whole-card plan. The hyperparameters,
//! every tensor's role, the typed description and each layer's store bytes
//! are read once ([`describe`]; [`read`] refuses a file whose features no
//! program runs, every one listed); the plan itself is the hybrid trunk's
//! ([`PlanInputs::plan`]), which both Nemotron-H families share.
//!
//! A sequence's stores are an attention layer's K and V planes, which grow
//! with the context, and a Mamba-2 layer's scan state and conv ring, which do
//! not; an MLP layer holds nothing ([`Holds`]).

use gguf::Split;

use super::hparams::{Hparams, Kind};
use super::spec;
pub use crate::arch::hybrid::{Holds, KvLayout, PlaceError, PlanInputs};
use crate::placement::PlacementError;

/// `split`'s hyperparameters, then its tensors' roles, then its description,
/// the first that fails being the error; then a file with a feature the
/// engine does not run ([`PlacementError::Unimplemented`], every one listed).
pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
    describe(split)?.runnable()
}

/// [`read`] without the refusals: the file as it is, for a gate that plans or
/// describes a file whatever the coverage list holds.
pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
    let hp = Hparams::read(split)?;
    let read = spec::read(split)?;
    PlanInputs::new(read.tensors, read.spec, kv_layout(&hp), 0)
}

/// The store each layer of `hp` holds: a Mamba-2 layer its scan, an attention
/// layer its key heads' planes, an MLP layer nothing.
#[must_use]
pub fn kv_layout(hp: &Hparams) -> KvLayout {
    KvLayout::of(
        &hp.ssm,
        hp.key_len,
        hp.kinds
            .iter()
            .zip(&hp.kv)
            .map(|(kind, &kv_heads)| match kind {
                Kind::Mamba => Holds::Scan,
                Kind::Attention => Holds::Attention { kv_heads },
                Kind::Mlp => Holds::Nothing,
            }),
    )
}

#[cfg(test)]
mod tests {
    use gguf::Split;

    use super::super::spec::tests::{CONV, HEAD, INNER, KV, geo, keys, tensors};
    use super::{PlaceError, PlanInputs, describe};
    use crate::arch::hybrid::Experts;
    use crate::arch::synthetic::{V, header_shaped};
    use crate::placement::workstation;
    use crate::placement::{KvBytes, PlanLevers};

    /// The synthetic header's layers, in order: Mamba, MLP, attention, and
    /// the same again.
    const KINDS: &str = "mpampa";

    fn inputs(tag: &str) -> PlanInputs {
        let g = geo();
        let path = header_shaped(
            tag,
            "nemotron_h",
            &keys(&g),
            &[
                ("tokenizer.ggml.pre", V::Str("qwen2")),
                ("tokenizer.chat_template", V::Str("{{ messages }}")),
            ],
            &tensors(&g, true),
        );
        let split = Split::open(&path).expect("the synthetic header opens");
        let inputs = describe(&split).expect("the synthetic header describes");
        let _ = std::fs::remove_file(&path);
        inputs
    }

    /// The conv channels and the scan state the synthetic layers hold.
    fn mamba_bytes() -> u64 {
        let g = geo();
        let conv_dim = INNER + 2 * g.n_groups * g.d_state;
        (INNER * g.d_state + (CONV - 1 + 8) * conv_dim) * 4
    }

    /// An attention layer's bytes grow with the context by K and V of its
    /// heads; a Mamba-2 layer's are one number at every context; an MLP layer
    /// and a layer past the last hold none.
    #[test]
    fn each_kind_holds_its_own_bytes() {
        let kv = inputs("nemotron-place-bytes").kv;
        let row = 2 * (KV as u64 * HEAD) * 2;
        assert_eq!(row, 64);
        assert_eq!(mamba_bytes(), 62_848);
        for (layer, kind) in KINDS.chars().enumerate() {
            for ctx in [1, 9, 4096] {
                let want = match kind {
                    'm' => mamba_bytes(),
                    'a' => row * ctx,
                    _ => 0,
                };
                assert_eq!(kv.layer_bytes(layer, ctx), want, "layer {layer} at {ctx}");
            }
        }
        assert_eq!(kv.layer_bytes(KINDS.len(), 9), 0);
        let sum: u64 = (0..KINDS.len()).map(|l| kv.layer_bytes(l, 9)).sum();
        assert_eq!(sum, 2 * mamba_bytes() + 2 * row * 9);
    }

    /// Two plans of one file at two contexts differ in the cache term by the
    /// attention layers' bytes of the extra positions and by nothing else:
    /// the Mamba-2 state is not counted a position.
    #[test]
    fn the_context_moves_only_the_attention_term() {
        let inputs = inputs("nemotron-place-ctx");
        let machine = workstation::plan_gate(inputs.model.layers);
        let levers = PlanLevers::default();
        let (a, b) = (1_024u64, 4_096u64);
        let small = inputs.plan(&machine, a, &levers).expect("a short plan");
        let large = inputs.plan(&machine, b, &levers).expect("a long plan");
        let row = 2 * (KV as u64 * HEAD) * 2;
        let attn_layers = 2;
        assert_eq!(
            large.cards[0].kv_bytes - small.cards[0].kv_bytes,
            attn_layers * row * (b - a)
        );
        assert_eq!(
            small.cards[0].kv_bytes,
            2 * mamba_bytes() + attn_layers * row * a
        );
        assert_eq!(small.cards[0].dense_bytes, large.cards[0].dense_bytes);
        assert_eq!(small.host.experts + small.host.expert_bytes, 0);
        assert_eq!(small.nvme_bytes, 0);
    }

    /// A file with no routed stacks has no expert to place: every arm of
    /// [`Experts`] plans it as [`PlanInputs::plan`] does.
    #[test]
    fn a_file_with_no_routed_stacks_plans_the_same_under_every_arm() {
        let inputs = inputs("nemotron-place-arms");
        let machine = workstation::plan_gate(inputs.model.layers);
        let levers = PlanLevers::default();
        let card = inputs.plan(&machine, 4, &levers).expect("a plan");
        for experts in [Experts::Card, Experts::Placed, Experts::Host] {
            let plan = inputs
                .plan_with(&machine, 4, &levers, experts)
                .unwrap_or_else(|e| panic!("{experts:?}: {e}"));
            assert_eq!(format!("{plan:?}"), format!("{card:?}"), "{experts:?}");
        }
    }

    /// A context of no position, a tier card and a second card are refused by
    /// name.
    #[test]
    fn a_machine_or_context_the_body_does_not_run_is_refused() {
        let inputs = inputs("nemotron-place-refuse");
        let levers = PlanLevers::default();
        let machine = workstation::plan_gate(inputs.model.layers);
        assert!(matches!(
            inputs.plan(&machine, 0, &levers),
            Err(PlaceError::CtxZero)
        ));
        let mut tiered = machine.clone();
        let mut tier = tiered.cards[0].clone();
        tier.layers = 0..0;
        tier.head = false;
        tier.token_embedding = false;
        tiered.tiers.push(tier);
        match inputs.plan(&tiered, 4, &levers) {
            Err(e @ PlaceError::Tier { tier: 0, .. }) => assert!(
                e.to_string()
                    .contains("the nemotron_h body runs every layer on one card"),
                "{e}"
            ),
            other => panic!("{other:?}, not the tier refusal"),
        }
        let mut two = machine;
        let mut second = two.cards[0].clone();
        second.layers = 0..0;
        second.head = false;
        second.token_embedding = false;
        two.cards.push(second);
        match inputs.plan(&two, 4, &levers) {
            Err(e @ PlaceError::Cards(2, _)) => assert_eq!(
                e.to_string(),
                "the placement puts the layers on 2 cards; the nemotron_h chain runs on one"
            ),
            other => panic!("{other:?}, not the card-count refusal"),
        }
    }
}
