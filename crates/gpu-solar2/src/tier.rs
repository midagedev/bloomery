//! The Solar-Open2 expert tier ([`bloomery_gpu::host::leg_tier`]): a card
//! beside the stage card that holds some of each routed layer's experts and
//! computes the routed slots the slot map sends it, in the host leg's shadow.
//! The tier's layer is the common [`bloomery_gpu::host::leg_tier::LegTier`]
//! over Solar's leg shape ([`CHUNK`] tokens a run), the
//! stage card's side the common
//! [`TierSide`](bloomery_gpu::host::leg_tier::TierSide) and its join the
//! routed block's ([`crate::ffn::Routed`]); what is Solar's here is the node
//! family that reads the file's description ([`NODE_FAMILY`]).

use std::ops::Range;

use bloomery_gpu::COL_GROUP;
use bloomery_gpu::GpuError;
use bloomery_gpu::host::leg_tier::{self, LegFamily, LegWidths};
use bloomery_gpu::host::node::NodeFamily;
use gguf::Split;
use model::ModelError;
use model::arch::solar2::host as solar_host;
use model::arch::solar2::place::{self, PlanInputs};
use model::moe::HostLayer;
use model::placement::ModelTensors;
use model::r8file::R8Source;
use models::LayerSpec;

use crate::body::LOAD_ROWS;

/// What the tier's errors name.
const WHAT: &str = "solar2 tier";

/// Tokens one tier run covers: the m-column launch group.
pub const CHUNK: usize = COL_GROUP;

// The plan reserves the tier's chunk scratch at the body's chunk.
const _: () = assert!(place::TIER_CHUNK as usize == CHUNK);

/// Solar's architecture facts for the common tier node service: the file's
/// reading, the rows in flight of a load ([`LOAD_ROWS`]) and the tier's chunk.
struct Solar2;

impl LegFamily for Solar2 {
    const WHAT: &'static str = WHAT;
    const WORD: &'static str = "solar2";
    const ROWS: usize = LOAD_ROWS;
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
        let experts = &inputs.hp.experts;
        LegWidths {
            n_expert: experts.n_expert,
            n_used: experts.n_used,
            n_embd: inputs.hp.n_embd,
            ff: experts.ff,
            program: model::placement::accepts::Program::Solar2Body,
        }
    }

    fn routed(
        src: R8Source<'_>,
        inputs: &PlanInputs,
        run: Range<usize>,
    ) -> Result<Vec<HostLayer>, ModelError> {
        solar_host::routed_layers(src, &inputs.hp, &inputs.spec, run)
    }
}

/// Solar's tier node service ([`leg_tier::node_family`]).
pub const NODE_FAMILY: NodeFamily = leg_tier::node_family::<Solar2>();
