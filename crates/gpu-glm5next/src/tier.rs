//! The GLM-5.3-Flash expert tier ([`bloomery_gpu::host::leg_tier`]): a card
//! beside the stage card that holds some of each routed layer's experts and
//! computes the routed slots the slot map sends it, in the host leg's shadow.
//!
//! The stage card's side is the common
//! [`TierSide`](bloomery_gpu::host::leg_tier::TierSide). The row's tier image
//! carries the normed activation in f32: GLM's `_sel` entries read the q8_1
//! planes of Walk A, which the stage card makes in the shadow, after the go;
//! the tier quantizes the same values with the same kernel.
//!
//! The tier's own layer is the common
//! [`LegTier`](bloomery_gpu::host::leg_tier::LegTier) over GLM's leg shape:
//! [`CHUNK`] tokens a run. What is GLM's here is the node family
//! ([`NODE_FAMILY`]), which reads the file's description.

use std::ops::Range;

use bloomery_gpu::GpuError;
use bloomery_gpu::host::leg_tier::{self, LegFamily, LegWidths};
use bloomery_gpu::host::node::NodeFamily;
use gguf::Split;
use model::ModelError;
use model::arch::glm5next::place::PlanInputs;
use model::moe::HostLayer;
use model::placement::ModelTensors;
use model::r8file::R8Source;
use models::LayerSpec;

use crate::body::prefill::CHUNK;

/// What the tier's errors name.
const WHAT: &str = "glm5next tier";

/// GLM's architecture facts for the common tier node service: the file's
/// reading, the rows in flight of a load ([`crate::LOAD_ROWS`]) and the
/// tier's chunk.
struct Glm;

impl LegFamily for Glm {
    const WHAT: &'static str = WHAT;
    const WORD: &'static str = "glm5next";
    const ROWS: usize = crate::LOAD_ROWS;
    const CHUNK: usize = CHUNK;
    type Inputs = PlanInputs;

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
            program: model::placement::accepts::Program::Glm5nextBody,
        }
    }

    fn routed(
        src: R8Source<'_>,
        inputs: &PlanInputs,
        run: Range<usize>,
    ) -> Result<Vec<HostLayer>, ModelError> {
        crate::host::routed_layers(src, &inputs.hp, run)
    }
}

/// GLM's tier node service ([`leg_tier::node_family`]).
pub const NODE_FAMILY: NodeFamily = leg_tier::node_family::<Glm>();

// The plan reserves the tier's chunk scratch at the body's chunk.
const _: () = assert!(model::arch::glm5next::place::TIER_CHUNK as usize == CHUNK);
