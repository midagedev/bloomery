//! The Mistral Large 4 expert tier ([`bloomery_gpu::host::leg_tier`]): a card
//! beside the stage card that holds some of each routed layer's experts and
//! computes the routed slots the slot map sends it, in the host leg's shadow.
//! The tier's layer is the common
//! [`LegTier`](bloomery_gpu::host::leg_tier::LegTier) over the family's leg
//! shape ([`CHUNK`] tokens a run), the stage card's side the common
//! [`TierSide`](bloomery_gpu::host::leg_tier::TierSide) and its join the
//! routed block's ([`crate::ffn::Routed`]); what is Mistral Large 4's here is
//! the node family that reads the file's description ([`NODE_FAMILY`]).

use std::ops::Range;

use crate::body::rows::PASS_ROWS;
use bloomery_gpu::COL_GROUP;
use bloomery_gpu::GpuError;
use bloomery_gpu::card_leg::ExpertInput;
use bloomery_gpu::host::leg_tier::{self, LegFamily, LegWidths};
use bloomery_gpu::host::node::NodeFamily;
use gguf::Split;
use model::ModelError;
use model::arch::mistrallarge4::host;
use model::arch::mistrallarge4::place::{self, PlanInputs};
use model::arch::models::LayerSpec;
use model::moe::HostLayer;
use model::placement::ModelTensors;
use model::placement::accepts::Program;
use model::r8file::R8Source;

/// What the tier's errors name.
const WHAT: &str = "mistrallarge4 tier";

/// Tokens one tier run covers: the m-column launch group.
pub const CHUNK: usize = COL_GROUP;

// The plan reserves the tier's chunk scratch at the body's chunk.
const _: () = assert!(place::TIER_CHUNK as usize == CHUNK);

/// Mistral Large 4's architecture facts for the common tier node service: the
/// file's reading, the rows a load holds in flight (the step's and a
/// verify's, [`PASS_ROWS`]), the tier's chunk, and the expert input the stage
/// card's routed block runs ([`crate::ffn::CONFIG`]).
struct MistralLarge4;

impl LegFamily for MistralLarge4 {
    const WHAT: &'static str = WHAT;
    const WORD: &'static str = "mistrallarge4";
    const ROWS: usize = PASS_ROWS;
    const CHUNK: usize = CHUNK;
    type Inputs = PlanInputs;

    fn expert_input(_inputs: &PlanInputs) -> ExpertInput {
        crate::ffn::CONFIG.expert_input
    }

    fn describe(file: &Split) -> Result<PlanInputs, GpuError> {
        PlanInputs::describe(file).map_err(|e| GpuError::plan(WHAT, e))
    }

    fn into_model(inputs: PlanInputs) -> ModelTensors {
        inputs.model
    }

    fn layers(inputs: &PlanInputs) -> &[LayerSpec] {
        &inputs.spec.layers
    }

    fn widths(inputs: &PlanInputs) -> LegWidths {
        let hp = &inputs.hp;
        LegWidths {
            n_expert: hp.n_expert,
            n_used: hp.n_used,
            n_embd: hp.n_embd,
            ff: hp.expert_ff,
            program: Program::MistralLarge4Body,
        }
    }

    fn routed(
        src: R8Source<'_>,
        inputs: &PlanInputs,
        run: Range<usize>,
    ) -> Result<Vec<HostLayer>, ModelError> {
        host::routed_layers(src, &inputs.hp, &inputs.spec, run)
    }
}

/// Mistral Large 4's tier node service ([`leg_tier::node_family`]).
pub const NODE_FAMILY: NodeFamily = leg_tier::node_family::<MistralLarge4>();
