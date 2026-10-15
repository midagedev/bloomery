//! A `nemotron_h` file planned onto a machine: the one path from the file's
//! headers to a whole-card plan. The hyperparameters, every tensor's role,
//! the typed description and each layer's store bytes ([`KvLayout`]) are read
//! once ([`PlanInputs::read`], which refuses a file whose features no program
//! runs, every one listed); then the placement ([`PlanInputs::plan`]) puts
//! the whole model on one card (the file holds no routed expert, so no tier
//! and no host leg has anything to hold), and refuses by name a machine of
//! more than one card or with a tier card, a context of no position, and a
//! context whose stores the card cannot hold.
//!
//! A sequence's stores are an attention layer's K and V planes, which grow
//! with the context, and a Mamba-2 layer's scan state and conv ring, which do
//! not; an MLP layer holds nothing ([`KvLayout::layer_bytes`]).

use gguf::Split;
use models::ModelSpec;
use runtime::stores::{dense_kv_bytes, mamba2_bytes};

use super::hparams::{Hparams, Kind};
use super::spec;
use crate::arch::coverage;
use crate::placement::accepts::{Program, routed_rule};
use crate::placement::ctx_ask::{PlanRefusal, Short};
use crate::placement::{
    self, KvBytes, Machine, ModelTensors, PlacementError, Plan, PlanLevers, Unimplemented,
    Violation, checked, joined,
};

/// What a plan of a `nemotron_h` file is made from, read from its headers.
#[derive(Clone, Debug)]
pub struct PlanInputs {
    /// Every tensor with its role, and the model's layer count.
    pub model: ModelTensors,
    /// The typed model description the coverage check reads.
    pub spec: ModelSpec,
    /// Each layer's store bytes.
    pub kv: KvLayout,
}

/// Why a plan was refused.
#[derive(Debug, thiserror::Error)]
pub enum PlaceError {
    /// The placement could not be built.
    #[error(transparent)]
    Placement(#[from] PlacementError),
    /// A machine with an expert tier card: the file has no routed expert, so
    /// the tier would hold nothing.
    #[error(
        "the machine has the expert tier card {card} (tier {tier}); the nemotron_h file has no routed expert for a tier to hold"
    )]
    Tier {
        /// The tier card's name.
        card: String,
        /// Its index among the machine's tiers.
        tier: usize,
    },
    /// A machine of other than one card: the body runs its chain on one.
    #[error("the placement puts the layers on {0} cards; the nemotron_h chain runs on one")]
    Cards(usize),
    /// The plan was built and breaks these invariants, every one of them.
    #[error("the plan breaks its invariants: {}", joined(.0))]
    Broken(Vec<Violation>),
    /// A context of no position: no plane, nothing to serve.
    #[error("ctx_max 0: a plan serves at least one position")]
    CtxZero,
}

impl PlanRefusal for PlaceError {
    fn placement(&self) -> Option<&PlacementError> {
        match self {
            Self::Placement(e) => Some(e),
            _ => None,
        }
    }

    fn violations(&self) -> Option<&[Violation]> {
        match self {
            Self::Broken(v) => Some(v),
            _ => None,
        }
    }

    /// The card's reading alone: the plan has no host side, so a context that
    /// does not fit is short of the card's bytes.
    fn short(&self) -> Option<Short> {
        self.card_short()
    }
}

impl PlanInputs {
    /// `split`'s hyperparameters, then its tensors' roles, then its
    /// description, the first that fails being the error; then a file with a
    /// feature the engine does not run ([`PlacementError::Unimplemented`],
    /// every one listed).
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        let inputs = PlanInputs::describe(split)?;
        let missing = inputs.unimplemented();
        if missing.is_empty() {
            Ok(inputs)
        } else {
            Err(PlacementError::Unimplemented(missing))
        }
    }

    /// [`PlanInputs::read`] without the refusals: the file as it is, for a
    /// gate that plans or describes a file whatever the coverage list holds.
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let read = spec::read(split)?;
        let kv = KvLayout::of(&hp);
        Ok(PlanInputs {
            model: read.tensors,
            spec: read.spec,
            kv,
        })
    }

    /// Every feature of the file the engine does not run, as
    /// [`coverage::check`] lists them. Empty for a file the program runs.
    pub fn unimplemented(&self) -> Vec<Unimplemented> {
        coverage::check(&self.spec, &self.model)
    }

    /// The placement of the file on `machine` at `ctx_max` positions under
    /// the placement's `levers`: the whole model on the machine's one card.
    pub fn plan<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
    ) -> Result<Plan<'a>, PlaceError> {
        if ctx_max == 0 {
            return Err(PlaceError::CtxZero);
        }
        if let Some((tier, card)) = machine.tiers.iter().enumerate().next() {
            return Err(PlaceError::Tier {
                card: card.name.clone(),
                tier,
            });
        }
        if machine.cards.len() != 1 {
            return Err(PlaceError::Cards(machine.cards.len()));
        }
        let plan = placement::plan_routed(
            &self.model,
            machine,
            ctx_max,
            &self.kv,
            levers,
            routed_rule(Program::NemotronHBody),
        )?;
        checked(plan).map_err(PlaceError::Broken)
    }
}

/// What one layer holds per sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Store {
    /// An attention layer: K and V of this many bytes a position.
    Rows(u64),
    /// A Mamba-2 layer: this many bytes of scan state and conv ring, whatever
    /// the context.
    State(u64),
    /// An MLP layer.
    Nothing,
}

/// Each layer's store bytes of a `nemotron_h` file: the bytes the body's
/// allocations hold ([`runtime::stores`] is their owner).
#[derive(Clone, Debug)]
pub struct KvLayout {
    layers: Vec<Store>,
}

impl KvLayout {
    /// The layout of the layers `hp` names.
    #[must_use]
    pub fn of(hp: &Hparams) -> KvLayout {
        let ssm = hp.ssm;
        let state = mamba2_bytes(ssm.d_inner, ssm.d_state, ssm.d_conv, ssm.conv_dim);
        let layers = hp
            .kinds
            .iter()
            .zip(&hp.kv)
            .map(|(kind, &kv_heads)| match kind {
                Kind::Mamba => Store::State(state),
                Kind::Attention => Store::Rows(dense_kv_bytes(kv_heads, hp.key_len, 1)),
                Kind::Mlp => Store::Nothing,
            })
            .collect();
        KvLayout { layers }
    }
}

impl KvBytes for KvLayout {
    /// An attention layer's K and V of `ctx_max` positions, a Mamba-2
    /// layer's state, an MLP layer's nothing; 0 past the last layer.
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        match self.layers.get(layer) {
            Some(Store::Rows(row)) => row.saturating_mul(ctx_max),
            Some(Store::State(bytes)) => *bytes,
            Some(Store::Nothing) | None => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use gguf::Split;

    use super::super::spec::tests::{CONV, HEAD, INNER, KV, geo, keys, tensors};
    use super::{KvBytes, PlaceError, PlanInputs, PlanLevers};
    use crate::arch::synthetic::{V, header_shaped};
    use crate::placement::workstation;

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
        let inputs = PlanInputs::describe(&split).expect("the synthetic header describes");
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
                    .contains("no routed expert for a tier to hold"),
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
            Err(e @ PlaceError::Cards(2)) => assert_eq!(
                e.to_string(),
                "the placement puts the layers on 2 cards; the nemotron_h chain runs on one"
            ),
            other => panic!("{other:?}, not the card-count refusal"),
        }
    }
}
