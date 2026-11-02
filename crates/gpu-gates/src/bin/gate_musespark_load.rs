//! The muse-spark load gate: the body's placed load of the gate plan, proven
//! against the plan. The plan is the gate card's (`gate_card`: the 3090's
//! bytes on the card in view in the real tier, the largest visible card in the
//! fixture tier, which under a budget lever keeps half of its experts on the
//! card so the host share is not empty), one stage card running every layer
//! and the head with no expert tier, at [`placed_load::CTX`] positions. The body is opened
//! through the entry the engine opens the family with
//! (`GpuModel::<Body>::open_placed`). A correctness run: every time and byte
//! figure it prints is a runtime value. Every clause is self-consistency (the
//! load against its own plan and the file), so both tiers run all five.
//!
//! - `plan_places_every_expert`: the plan puts each routed expert in exactly
//!   one place — the card's prefix and the host's rest, no tier and no paged
//!   expert — the card's expert bytes are its experts' stacks in the card
//!   format's buffers (the loader's arithmetic, which pads each row to whole
//!   words), the host's are its experts' file bytes, and the card's KV term is
//!   the typed description's planes.
//! - `segments_are_the_plans`: every segment the plan puts on the card is
//!   resident at the plan's buffer bytes (measured from the buffers), nothing
//!   else is resident, and the weights' total is the card's dense and expert
//!   bytes.
//! - `expert_prefix_byte_for_byte`: on a sample of layers (the first, the
//!   middle and the last), each routed stack on the card is the file's own
//!   bytes of the first `n_l` experts, row count included; a sample of the
//!   F32 norms is the file's values bit for bit.
//! - `slot_map_is_the_prefix`: each layer's row of the tier's slot map holds
//!   the card's experts at slots `0..n_l` in id order and the host mark on
//!   the rest, and the layer's card count is the plan's.
//! - `resident_inside_the_plan`: what the load holds on the card (the
//!   weights, the K/V planes, the step's buffers, the host boundary, the
//!   prompt pass's buffers) stays inside the plan's terms for them, the
//!   prompt reserve among them.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_musespark_load", gate::run())
}

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the card the gate plans on; the fixture arm reads its spec"
)]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "gpu")]
#[path = "shared/placed_load.rs"]
mod placed_load;

#[cfg(feature = "gpu")]
mod gate {
    use crate::gate_card;
    use crate::placed_load::{self, CTX, Clause, HOST_CFG, Routed, STACKS};
    use bloomery_gpu::GpuModel;
    use bloomery_gpu::arch::musespark::Body;
    use bloomery_gpu::host::served::TierBody;
    use bloomery_gpu::weights::DevWeight;
    use bloomery_gpu_gates::tier;
    use bloomery_gpu_gates::{GateError, open_split};
    use gguf::quant::GgmlType;
    use model::arch::Arch;
    use model::arch::musespark::hparams::Hparams;
    use model::arch::musespark::place::PlanInputs;
    use model::placement::{Plan, workstation};

    type Model = GpuModel<Body>;

    const NAME: &str = "gate_musespark_load";

    /// The F32 norms whose values are compared bit for bit, per sampled layer.
    const NORMS: [&str; 3] = ["attn_norm", "ffn_post_exp_norm", "post_ffw_norm"];

    fn expert_prefix_byte_for_byte(
        m: &Model,
        plan: &Plan<'_>,
        file: &gguf::Split,
        shape: Routed,
    ) -> Result<bool, GateError> {
        const CLAUSE: &str = "expert_prefix_byte_for_byte";
        let stream = m.gpu().stream();
        let w = m.weights();
        let (mut stacks, mut norms, mut bad) = (0usize, 0usize, Vec::new());
        for l in shape.sample() {
            let n_l = usize::try_from(plan.n_l[l])?;
            for kind in STACKS {
                let name = format!("blk.{l}.ffn_{kind}_exps.weight");
                let (shard, info) = file
                    .find(&name)
                    .ok_or_else(|| format!("{name} is not in the file"))?;
                if info.ty != GgmlType::IQ2_XXS || info.dims.len() != 3 {
                    bad.push(format!(
                        "{name}: {:?} {:?}, want a IQ2_XXS stack",
                        info.ty, info.dims
                    ));
                    continue;
                }
                let per = usize::try_from(info.nbytes)? / shape.n_expert;
                let rows_per = usize::try_from(info.dims[1])?;
                let resident = w.get(&name);
                if n_l == 0 {
                    if resident.is_some() {
                        bad.push(format!(
                            "{name}: resident, and the plan keeps no expert of the layer"
                        ));
                    }
                    continue;
                }
                let Some(DevWeight::KQuant { ty, w: words, .. }) = resident else {
                    bad.push(format!("{name}: not resident as a K-quant stack"));
                    continue;
                };
                let got = words.buf().to_host_vec(stream)?;
                let bytes: Vec<u8> = got.iter().flat_map(|x| x.to_le_bytes()).collect();
                let data = file
                    .shard(shard)
                    .ok_or("shard index out of range")?
                    .data(info)?;
                let want = &data[..n_l * per];
                stacks += 1;
                if *ty != GgmlType::IQ2_XXS
                    || words.rows() != n_l * rows_per
                    || bytes.get(..n_l * per) != Some(want)
                {
                    bad.push(format!(
                        "{name}: {ty:?}, {} rows resident (want {}), first {} B {} the file's \
                         first {n_l} experts",
                        words.rows(),
                        n_l * rows_per,
                        n_l * per,
                        if bytes.get(..n_l * per) == Some(want) {
                            "equal"
                        } else {
                            "differ from"
                        }
                    ));
                }
            }
            for stem in NORMS {
                let name = format!("blk.{l}.{stem}.weight");
                if placed_load::check_f32(file, w, stream, &name, &mut bad)? {
                    norms += 1;
                }
            }
        }
        let text = format!(
            "layers {:?}, {stacks} resident stacks and {norms} norms read back",
            shape.sample()
        );
        Ok(placed_load::report(CLAUSE, &text, &bad))
    }

    fn slot_map_is_the_prefix(
        m: &Model,
        plan: &Plan<'_>,
        shape: Routed,
    ) -> Result<bool, GateError> {
        let placed = m
            .body(NAME)?
            .placed()
            .ok_or("a placed load carries the host tier")?;
        placed_load::slot_map_is_the_prefix(placed.hybrid().slots(), plan, shape)
    }

    pub fn run() -> Result<(), GateError> {
        let tier_now = tier::Tier::from_env()?;
        let file = open_split(Arch::MuseSpark, "gate-gpu-musespark-load")?;
        let hp = Hparams::read(&file).map_err(|e| format!("muse-spark header: {e}"))?;
        let shape = Routed {
            n_layer: hp.n_layer,
            dense_lead: 0,
            n_expert: hp.n_expert,
        };
        let inputs = PlanInputs::read(&file)?;
        let spec = gate_card::card()?;
        let machine = workstation::placed_on(spec, hp.n_layer, workstation::SCRATCH);
        let levers = placed_load::levers(&inputs, &machine, tier_now)?;
        let plan = inputs.plan(&machine, CTX, &levers)?;
        let mut ok = placed_load::run_clause("plan_places_every_expert", || {
            placed_load::plan_places_every_expert(&plan, &inputs, shape)
        })?;
        let t = std::time::Instant::now();
        let m = Model::open_placed(
            open_split(Arch::MuseSpark, "gate-gpu-musespark-load")?,
            &plan,
            0,
            HOST_CFG,
        )?;
        placed_load::load_line(m.resident_bytes(), m.layers().len(), t.elapsed());
        let clauses: Vec<(&str, Clause<'_>)> = vec![
            (
                "segments_are_the_plans",
                Box::new(|| placed_load::segments_are_the_plans(&plan, m.weights())),
            ),
            (
                "expert_prefix_byte_for_byte",
                Box::new(|| expert_prefix_byte_for_byte(&m, &plan, &file, shape)),
            ),
            (
                "slot_map_is_the_prefix",
                Box::new(|| slot_map_is_the_prefix(&m, &plan, shape)),
            ),
            (
                "resident_inside_the_plan",
                Box::new(|| {
                    placed_load::resident_inside_the_plan(
                        m.resident_bytes() as u64,
                        &plan,
                        inputs.prompt_reserve,
                    )
                }),
            ),
        ];
        // Every clause is self-consistency, run in both tiers; the first read
        // the plan alone, before the load.
        ok &= placed_load::run_clauses(&clauses)?;
        placed_load::conclude(ok)?;
        println!(
            "PASSED: {NAME} the plan puts every routed expert in one place and its bytes are the \
             file's; every segment on the card is resident at the plan's buffers and nothing \
             else is; the routed stacks and the sampled norms are the file's bytes; the slot \
             map is the ids' prefix; the load stays inside the plan"
        );
        Ok(())
    }
}
