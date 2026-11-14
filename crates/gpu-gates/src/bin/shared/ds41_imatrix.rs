//! The V4.1 body as the importance collector's walk (`bloomery_gpu_gates::imatrix_walk`): one chunk is the
//! model from empty through one prompt call, the prompt batch (`body::prefill`), whose launches the collector's
//! feeds sit in. `ppl_ds41 --imatrix` (the tool) walks a session's model with it, and `gate_deepseek41_prefill
//! --imatrix` walks the gate's own load, so one prompt path answers both the tool and its gate.
//!
//! A collecting walk needs three facts of the load, made by [`ready`] and refused by name: the prompt is fed as
//! batches (the decode step's captured launches hold no feed), every layer runs every position (the CED
//! triangle leaves positions out of a layer, so its counts would fall short of the rows `llama-imatrix` feeds),
//! and the card's routed experts run on the tile path, which holds the SwiGLU rows as f32 between the gate·up
//! and the down (the grouped GEMM quantizes them inside its SwiGLU launch). The tile path is the steps' bits.

use app::Prompt;
use bloomery_gpu_deepseek41::body::{Body, CedState, Deepseek41Model, PrefillMode};
use bloomery_gpu_gates::GateError;

/// What every refusal of this walk names.
const WHAT: &str = "ds41_imatrix";

/// One chunk of the collecting walk (`imatrix_walk::collect`'s `run`): the engine's reset and the residency
/// machine back to its seed, as `Session::clear` of one slot does, then the chunk as one prompt call. The chunk runs
/// from an empty model whatever the placement moved before it.
pub fn chunk(m: &mut Deepseek41Model, ids: &[u32]) -> Result<(), GateError> {
    m.reset()?;
    m.residency_reset()?;
    Body::prompt(m, ids)?;
    Ok(())
}

/// The load made ready for the walk: the card's routed experts on the tile path, then [`check`].
pub fn ready(m: &mut Deepseek41Model) -> Result<(), GateError> {
    m.body_parts(WHAT)?.2.set_batch_card_tiles(true)?;
    check(m)
}

/// Refused by name before the first chunk: a body that feeds the prompt as steps (the file's batch refusal named),
/// and a load whose prompt triangle is on.
fn check(m: &Deepseek41Model) -> Result<(), GateError> {
    let body = m.body(WHAT)?;
    if body.prefill_mode() != PrefillMode::Batch {
        return Err(format!(
            "{WHAT}: the load feeds a prompt as steps ({}): the collector's feeds sit in the prompt batch's launches; \
             BLOOMERY_PREFILL=batch on a file whose tensors the batch reads",
            body.batch_refusal().unwrap_or("BLOOMERY_PREFILL=steps")
        )
        .into());
    }
    match body.ced() {
        CedState::Off(_) => Ok(()),
        CedState::On => Err(format!(
            "{WHAT}: the prompt triangle is on: a layer runs only the positions a later reader needs, so a weight's \
             counts fall short of the rows fed; BLOOMERY_CED=off"
        )
        .into()),
    }
}
