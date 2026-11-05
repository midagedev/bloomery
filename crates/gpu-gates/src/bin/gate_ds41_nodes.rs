//! Compare a V4.1 Q8 node host tier with the same build's local CPU tier.

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_ds41_nodes", gate::run())
}

#[cfg(feature = "deepseek41")]
#[path = "shared/node_history.rs"]
mod node_history;

#[cfg(feature = "deepseek41")]
mod gate {
    use crate::node_history::CTX;

    use bloomery_gpu::host::tier::TierOpen;
    use bloomery_gpu_deepseek41::body::{Body, BodyMeta, OpenCfg};
    use bloomery_gpu_gates::{GateError, ref_model_path};
    use bloomery_levers::Levers;
    use gguf::Split;
    use model::arch::deepseek41::place::{PlanInputs, tier_batch};
    use model::placement::PlanLevers;

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let (budget, out) = crate::node_history::args()?;
        let levers = Levers::from_env()?;
        let file = Split::open(ref_model_path()?)?;
        let inputs = PlanInputs::describe(&file)?;
        let (machine, node) =
            crate::node_history::machine(inputs.model.layers, tier_batch(&inputs.hp))?;
        let half = inputs.model.experts / 2;
        let plan = inputs.plan_nodes(
            &machine,
            CTX,
            &PlanLevers {
                card_budget_bytes: Some(budget).into(),
                ..PlanLevers::default()
            },
            &[half, inputs.model.experts - half],
        )?;
        let cfg = OpenCfg::from_levers(&levers)?;
        let meta = BodyMeta {
            hp: inputs.hp.clone(),
            levers: cfg.body,
        };
        let model =
            Body::open_placed_tiered(file, &plan, 0, TierOpen::of_machine(&machine)?, &meta)?;
        crate::node_history::write(model, &plan, &out, node)
    }
}
