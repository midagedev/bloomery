//! Compare a GLM node host tier with the same build's local CPU tier.

#[cfg(feature = "glm5next")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_glm5next_nodes", gate::run())
}

#[cfg(feature = "glm5next")]
#[path = "shared/node_history.rs"]
mod node_history;

#[cfg(feature = "glm5next")]
mod gate {
    use crate::node_history::CTX;

    use bloomery_gpu::host::swap::Residency;
    use bloomery_gpu_gates::residency38::{Seqs, reserve_checkpoints};
    use bloomery_gpu_gates::{GateError, ref_model_path};
    use bloomery_gpu_glm5next::{Body, checkpoint_price};
    use bloomery_levers::Levers;
    use gguf::Split;
    use model::arch::glm5next::place::{KdaLanes, PlanInputs, tier_batch};
    use model::placement::PlanLevers;

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let (budget, out) = crate::node_history::args()?;
        let levers = Levers::from_env()?;
        let path = ref_model_path()?;
        let file = Split::open(&path)?;
        let inputs = PlanInputs::describe(&file)?;
        let (mut machine, node) =
            crate::node_history::machine(inputs.model.layers, tier_batch(&inputs)?)?;
        reserve_checkpoints(
            &mut machine,
            Seqs::priced(1, checkpoint_price(&inputs)?),
            CTX,
        )?;
        let half = inputs.model.experts / 2;
        let plan = inputs.plan_nodes(
            &machine,
            CTX,
            &PlanLevers {
                card_budget_bytes: Some(budget),
            },
            KdaLanes::Two,
            &[half, inputs.model.experts - half],
        )?;
        let model = Body::open_placed_lanes(
            file,
            &plan,
            &inputs,
            0,
            levers.host(),
            Residency::Off,
            KdaLanes::Two,
        )?;
        crate::node_history::write(model, &plan, &out, node)
    }
}
