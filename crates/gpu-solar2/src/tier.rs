//! The Solar-Open2 expert tier ([`bloomery_gpu::host::leg_tier`]): a card
//! beside the stage card that holds some of each routed layer's experts and
//! computes the routed slots the slot map sends it, in the host leg's shadow.
//! The tier's layer is the common [`LegTier`] over Solar's leg shape
//! ([`leg_shape`]); what is Solar's here is the facts of that shape, the node
//! family that reads the file's description ([`NODE_FAMILY`]) and the open of
//! the tier card ([`open_tier`]).

use std::ops::Range;
use std::sync::Arc;

use bloomery_gpu::card_leg::LegShape;
use bloomery_gpu::host::leg_tier::{self, LegOpen, LegTier};
use bloomery_gpu::host::node::{NodeFamily, NodeOpen};
use bloomery_gpu::host::run::{HostRun, HostWidths};
use bloomery_gpu::host::tier::{TierCard, TierExperts, TierOpen};
use bloomery_gpu::hybrid::SlotMap;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{COL_GROUP, Gpu, GpuError};
use gguf::Split;
use model::ModelError;
use model::moe::HostLayer;
use model::placement::{ModelTensors, Plan};
use model::r8file::R8Source;
use models::LayerSpec;

/// What the tier's errors name.
const WHAT: &str = "solar2 tier";

/// Routed experts a layer holds.
pub const N_EXPERT: usize = 320;
/// Routed slots a token takes.
pub const N_USED: usize = 8;

/// Tokens one tier run covers: the m-column launch group.
pub const CHUNK: usize = COL_GROUP;

/// The rows in flight a load's tier page holds: one lane (the file has no
/// NextN head).
pub const LOAD_ROWS: usize = 1;

/// Solar's architecture facts for the common tier node service.
pub const NODE_FAMILY: NodeFamily = NodeFamily {
    word: "solar2",
    tensors: node_tensors,
    experts: node_experts,
    host: node_host,
};

/// What the node's family reads of the file: Solar's placement tensors, its
/// description and widths, and the host views of its routed layers. The
/// reader is the model side's (`model::arch::solar2`'s place and host
/// readers); this is the one call that waits for it.
#[allow(
    dead_code,
    reason = "built by the model side's reader that replaces `describe`"
)]
struct Described {
    tensors: ModelTensors,
    layers: Vec<LayerSpec>,
    n_embd: usize,
    expert_ff: usize,
    /// The host views of the routed layers `run`; a layer of the run that
    /// does not route is refused by name.
    routed: fn(R8Source<'_>, Range<usize>) -> Result<Vec<HostLayer>, ModelError>,
}

fn describe(file: &Split) -> Result<Described, GpuError> {
    let _ = file;
    unimplemented!("solar2 tier: the model side's place and host readers")
}

fn node_tensors(file: &Split) -> Result<ModelTensors, GpuError> {
    describe(file).map(|d| d.tensors)
}

fn node_described(file: &Split, open: &NodeOpen) -> Result<Described, GpuError> {
    let d = describe(file)?;
    LegOpen {
        n_expert: N_EXPERT,
        layers: d.layers.len(),
        hidden: d.n_embd,
        n_used: N_USED,
        rows: LOAD_ROWS,
    }
    .check(WHAT, open)?;
    Ok(d)
}

fn node_experts(
    gpu: &Gpu,
    file: &Split,
    _: &ModelTensors,
    w: &Weights,
    open: &NodeOpen,
) -> Result<Box<dyn TierExperts>, GpuError> {
    let d = node_described(file, open)?;
    Ok(Box::new(LegTier::new(
        gpu,
        &d.layers,
        &open.set,
        w,
        leg_shape(d.n_embd, d.expert_ff),
    )?))
}

fn node_host(file: Arc<Split>, open: &NodeOpen) -> Result<HostRun, GpuError> {
    let d = node_described(&file, open)?;
    let run = runtime::layer::hosted(&d.layers)
        .map_err(|e| GpuError::shape(WHAT, format!("the node host run: {e}")))?;
    HostRun::build(
        file,
        run.start,
        false,
        HostWidths {
            embd: d.n_embd,
            ff: d.expert_ff,
            n_used: N_USED,
        },
        |src| (d.routed)(src, run),
    )
}

/// The card leg's shape for the tier: up to [`CHUNK`] tokens of [`N_USED`]
/// slots, `n_embd` wide, through experts `ff` wide.
const fn leg_shape(n_embd: usize, ff: usize) -> LegShape {
    LegShape {
        cap: CHUNK,
        slots_per_col: N_USED,
        n: n_embd,
        ff,
    }
}

/// The tier card `t` of `plan`, tier `tier` of `map`, for the stage card
/// `stage`'s load, opened in this process or on its node through the common
/// owner ([`leg_tier::open_tier`], [`NODE_FAMILY`]): Solar's tier
/// computation over the map's rows of that tier (the description `layers`,
/// `n_embd` wide), its page rows `rows` ([`LOAD_ROWS`]); `card_dontneed` as
/// for the stage card's segments. The stage card's context is current again
/// on return. Load-time only.
#[allow(
    clippy::too_many_arguments,
    reason = "the stage card, the file, the plan, the tier and its index, the description, the map, the width and rows, and the page lever (rust-quality R8)"
)]
pub fn open_tier(
    stage: &Gpu,
    file: &Arc<Split>,
    plan: &Plan<'_>,
    t: &TierOpen,
    tier: usize,
    layers: &[LayerSpec],
    map: &SlotMap,
    [n_embd, rows]: [usize; 2],
    card_dontneed: bool,
) -> Result<TierCard, GpuError> {
    let open = LegOpen {
        n_expert: N_EXPERT,
        layers: layers.len(),
        hidden: n_embd,
        n_used: N_USED,
        rows,
    };
    leg_tier::open_tier(
        stage,
        file,
        plan,
        (t, tier),
        map,
        (&NODE_FAMILY, open),
        card_dontneed,
    )
}
