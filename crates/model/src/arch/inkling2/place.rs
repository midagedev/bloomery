//! An Inkling 2 file's placement inputs: the family's half of the path from the file's headers to
//! a plan. The tensors' roles, the typed description and the trunk's stores over its layer visits
//! ([`place_hybrid::KvLayout::of_visits`]) are read once ([`Inputs::describe`];
//! [`Inputs::read`] refuses a file whose features no program runs, every one listed — the
//! coverage list is the work queue of the rounds that build one). The plan itself
//! ([`Inputs::plan`]) is the host-routed one: no card kernel reads this family's routed stacks
//! yet, so every expert stays on the host and the card holds the dense weights, the visits' K and
//! V and their convolution states — the weights once per stored layer (a looped layer's passes
//! share them: the file holds one set), the stores once per visit. The engine that runs the card
//! side names its own program when it lands, and the plan follows it there.

use gguf::Split;
use models::ModelSpec;

use super::hparams::Hparams;
use super::spec;
use crate::arch::coverage;
use crate::placement::{
    Machine, ModelTensors, PlacementError, Plan, PlanLevers, Unimplemented, plan_host_routed,
};

pub use crate::arch::place_hybrid::KvLayout;

/// What a plan of an Inkling 2 file is made from: every tensor with its role, the typed
/// description the coverage check reads, and the trunk's stores over its layer visits.
#[derive(Clone, Debug)]
pub struct Inputs {
    /// The file's hyperparameters, carried for the body and never read here.
    pub hp: Hparams,
    /// Every tensor with its role, and the model's layer and expert counts.
    pub model: ModelTensors,
    /// The typed model description.
    pub spec: ModelSpec,
    /// The trunk's stores, a visit a slot ([`KvLayout::of_visits`]).
    pub kv: KvLayout,
}

impl Inputs {
    /// `split`'s tensors' roles, then its description, then its stores, the first that fails being
    /// the error.
    pub fn describe(split: &Split) -> Result<Inputs, PlacementError> {
        let read = spec::read(split)?;
        let hp = Hparams::read(split)?;
        let kv = KvLayout::of_visits(&read.spec)?;
        Ok(Inputs {
            hp,
            model: read.tensors,
            spec: read.spec,
            kv,
        })
    }

    /// [`Inputs::describe`] with the refusals: `Err` for a file with a feature no program runs,
    /// every one listed.
    pub fn read(split: &Split) -> Result<Inputs, PlacementError> {
        let inputs = Inputs::describe(split)?;
        let missing = inputs.unimplemented();
        if missing.is_empty() {
            Ok(inputs)
        } else {
            Err(PlacementError::Unimplemented(missing))
        }
    }

    /// Every feature of the file no program in this tree runs, as [`coverage::check`] lists them.
    /// Empty for a file a program runs.
    pub fn unimplemented(&self) -> Vec<Unimplemented> {
        coverage::check(&self.spec, &self.model)
    }

    /// `self`, or — for a file with a feature the engine does not run —
    /// [`PlacementError::Unimplemented`] with every one listed.
    pub fn runnable(self) -> Result<Inputs, PlacementError> {
        let missing = self.unimplemented();
        if missing.is_empty() {
            Ok(self)
        } else {
            Err(PlacementError::Unimplemented(missing))
        }
    }

    /// The placement of the file on `machine` at `ctx_max` positions under the placement's
    /// `levers`: every routed stack on the host — no card kernel reads this family's experts yet
    /// ([`placement::plan_host_routed`]) — the dense tensors where their roles say, and the
    /// visits' stores on the cards that run their layers. A machine that names an expert tier
    /// card is refused by name (the tier would hold nothing). The engine's own program replaces
    /// this plan when its card leg lands.
    pub fn plan<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
    ) -> Result<Plan<'a>, PlacementError> {
        plan_host_routed(&self.model, machine, ctx_max, &self.kv, levers)
    }
}

#[cfg(test)]
mod tests {
    use models::LoopSpec;

    use super::super::hparams::tests::{keys, tensors};
    use super::Inputs;
    use crate::arch::synthetic::{V, header_shaped};
    use crate::placement::{KvBytes, PlacementError, PlanLevers, workstation};

    /// The small header's file, opened; each test names its own file (the tests run in threads of
    /// one process). `typed` names a file whose routed stacks are Q8_0 (a type the host stack
    /// runs): the header-only writer's F32 default has no host kernel, so a plan of it is
    /// refused by the stack's name.
    fn opened(name: &str) -> gguf::Split {
        opened_typed(name, false)
    }

    /// [`opened`] with the routed stacks in Q8_0.
    fn opened_typed(name: &str, typed: bool) -> gguf::Split {
        let global = [
            ("tokenizer.ggml.pre", V::Str("inkling2")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let path = if typed {
            use crate::arch::synthetic::Ty;
            let tensors: Vec<(String, Vec<u64>, Ty)> = tensors(true)
                .into_iter()
                .map(|(n, d)| {
                    let ty = if n.ends_with("ffn_gate_up_exps.weight")
                        || n.ends_with("ffn_down_exps.weight")
                    {
                        Ty::Q8_0
                    } else {
                        Ty::F32
                    };
                    (n, d, ty)
                })
                .collect();
            crate::arch::synthetic::header_typed(name, "inkling2", &keys(true), &global, &tensors)
        } else {
            header_shaped(name, "inkling2", &keys(true), &global, &tensors(true))
        };
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let _ = std::fs::remove_file(&path);
        split
    }

    /// The described inputs of the small header: 6 stored layers, the core [2, 4) three passes,
    /// 10 visits — layers 2 and 3 twice more each — of 2 key heads of 16 (a K and V row of 128 B
    /// a position), every visit's conv state the widest layer's channels (2·32 + 2·64)·3 f32 =
    /// 2,304 B [derived from the header's widths].
    fn inputs(name: &str) -> Inputs {
        Inputs::describe(&opened(name)).expect("the synthetic header describes")
    }

    /// [`inputs`] of the typed header (the routed stacks Q8_0).
    fn typed_inputs(name: &str) -> Inputs {
        Inputs::describe(&opened_typed(name, true)).expect("the synthetic header describes")
    }

    /// The description's loop is the file's, and the stores are one a visit: the core's two
    /// layers hold their three visits' rows and states (3·(rows·128 + 2,304)), the other four
    /// layers one (rows·128 + 2,304), the windowed layers' rows on the ring. Mutant:
    /// the visits not counted (a core layer sized as its stem's).
    #[test]
    fn the_stores_are_one_a_visit() {
        let inputs = inputs("inkling2-place-visits");
        assert_eq!(
            inputs.spec.loop_,
            Some(LoopSpec {
                core_start: 2,
                core_end: 4,
                passes: 3,
                gate_rank: 8,
            })
        );
        let (row, conv) = (2 * 2 * 16 * 2, (2 * 32 + 2 * 64) * 3 * 4);
        assert_eq!((row, conv), (128, 2_304));
        let kv = &inputs.kv;
        // Layer 2 is windowless (its visits hold every position), layer 3 windowed, both core.
        assert_eq!(kv.layer_bytes(2, 64), 3 * (64 * row + conv));
        assert_eq!(
            kv.layer_bytes(3, 64),
            3 * (runtime_rows(16, 64) * row + conv)
        );
        assert_eq!(kv.layer_bytes(1, 64), runtime_rows(16, 64) * row + conv);
        assert_eq!(
            kv.layer_bytes(4, 64),
            64 * row + conv,
            "layer 4 is windowless"
        );
        assert_eq!(kv.layer_bytes(6, 64), 0, "past the trunk");
        // At 64 positions the ring holds every position, so each of the trunk's 10 visits holds
        // 64 rows and its state.
        assert_eq!(runtime_rows(16, 64), 64);
        let whole: u64 = (0..6).map(|l| kv.layer_bytes(l, 64)).sum();
        assert_eq!(whole, 10 * (64 * row + conv));
    }

    /// The ring's row count of a window of `window` at `ctx` positions
    /// ([`runtime::stores::ring_rows`], the layout's own rule).
    fn runtime_rows(window: u64, ctx: u64) -> u64 {
        runtime::stores::ring_rows(window, ctx)
    }

    /// The host-routed plan of the described file holds on the gate machine: the dense tensors
    /// and the visits' stores on the card, every routed stack on the host, and a machine that
    /// names an expert tier card is refused by name. The read still refuses the small header
    /// through the coverage check (its features are the work queue's).
    #[test]
    fn the_described_file_plans_with_its_experts_on_the_host() {
        let inputs = typed_inputs("inkling2-place-plan");
        let machine = workstation::plan_gate(6);
        let plan = inputs
            .plan(&machine, 64, &PlanLevers::default())
            .expect("the gate machine plans the file");
        assert!(
            plan.n_l.iter().all(|&n| n == 0),
            "no layer keeps experts on the card: {:?}",
            plan.n_l
        );
        assert!(plan.host.expert_bytes > 0, "the routed stacks are hosted");
        let tiered = workstation::plan_tiers(
            6,
            workstation::A6000,
            &[workstation::RTX_3090],
            None,
            workstation::NO_TIER_BATCH,
        )
        .expect("the tiered machine");
        let err = inputs
            .plan(&tiered, 64, &PlanLevers::default())
            .expect_err("a tier card is refused");
        assert!(err.to_string().contains("tier"), "{err}");
        match Inputs::read(&opened("inkling2-place-read")) {
            Err(PlacementError::Unimplemented(items)) => assert!(!items.is_empty()),
            other => panic!("{other:?}, not the coverage refusal"),
        }
    }
}
