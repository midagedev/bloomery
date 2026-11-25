//! The prompt pass: a prompt runs as units of up to the load's pass rows, each unit through every
//! layer at once (each part's prompt form over the unit's rows, every row the step's at its
//! position), the head on the last row of the last unit. On a load whose card holds every expert
//! each layer runs whole over the unit's rows; on one whose plan hosts experts a hosted layer
//! runs its three parts around the host tier's batch port, the unit's rows at once
//! ([`crate::placed::walk_pass`]), so each distinct host expert is read once for every row of the
//! unit that picked it. The call is one residency pass
//! ([`bloomery_gpu::host::served::prompt_pass`]); a load opened with no pass steps its ids.
//!
//! A call takes the checkpoints its marks name ([`bloomery_gpu::checkpoint::Checkpoints::marks`]):
//! its start and its end. A mark inside a call would cut it into passes at that position, one
//! after the other; the body's spacing names none, so a call is one pass.
//!
//! Under `BLOOMERY_XSTREAM=admit` a pass is one host streaming call ([`crate::stream`]): each unit's
//! walk picks the experts the card takes over.

use bloomery_gpu::host::served::prompt_pass;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError, GpuModel};

use crate::body::{Body, Run};
use crate::step;

/// What the pass's refusals name.
const CALL: &str = "ernie55::pass::prompt";

/// The prompt `tokens` from the model's position, and the checkpoints the call's marks name (its
/// start, its end); the argmax after the last. A checkpoint is taken after the pass has returned,
/// so a pass that faulted leaves none. Refused by name: no id, a call off the position the stores
/// stand at. A call that fails part-way leaves the stores where no step continues from: the next
/// call resets first.
pub fn prompt(model: &mut GpuModel<Body>, tokens: &[u32]) -> Result<u32, GpuError> {
    if tokens.is_empty() {
        return Err(GpuError::shape(CALL, "a prompt of no id"));
    }
    let from = model.pos();
    let to = u32::try_from(tokens.len())
        .ok()
        .and_then(|n| from.checked_add(n))
        .ok_or_else(|| GpuError::shape(CALL, "ids past u32"))?;
    let (_, _, body) = model.body_parts(CALL)?;
    body.ckpt.call(from);
    let marks = body.ckpt.marks(from, to);
    let mut token = None;
    let mut at = from;
    for mark in marks {
        if mark > at {
            let seg = &tokens[(at - from) as usize..(mark - from) as usize];
            token = Some(feed(model, seg)?);
            at = mark;
        }
        let pos = model.pos();
        let (gpu, _, body) = model.body_parts(CALL)?;
        body.checkpoint(gpu, pos)?;
    }
    token.ok_or_else(|| GpuError::shape(CALL, "a prompt that read no token"))
}

/// The prompt `tokens` from the model's position as the pass's units, or stepped on a load with no
/// pass; the argmax after the last. A waiting cut is carried out first.
fn feed(model: &mut GpuModel<Body>, tokens: &[u32]) -> Result<u32, GpuError> {
    if model.body(CALL)?.pass.is_none() {
        return model.step(tokens);
    }
    let n = u32::try_from(tokens.len()).map_err(|_| GpuError::shape(CALL, "ids past u32"))?;
    let token = prompt_pass(model, |model| {
        let start = model.pos();
        let history = {
            let (gpu, _, body) = model.body_parts(CALL)?;
            body.stream_begin(gpu.stream(), tokens.len(), start)?
        };
        let r = model.run_rows(tokens.len(), CALL, |gpu, w, body, head, pos| {
            body.apply_cut(gpu.stream())?;
            if pos != body.held {
                return Err(GpuError::shape(
                    CALL,
                    format!(
                        "a call at position {pos}; the stores stand at {} (reset after a call that \
                         failed part-way)",
                        body.held
                    ),
                ));
            }
            let rows = body.pass.ok_or(GpuError::State {
                what: CALL,
                missing: "the load's prompt pass",
            })?;
            body.held = u32::MAX;
            let units = tokens.len().div_ceil(rows);
            for (i, chunk) in tokens.chunks(rows).enumerate() {
                let at = pos
                    + u32::try_from(i * rows)
                        .map_err(|_| GpuError::shape(CALL, "a unit past u32"))?;
                unit((gpu, w), body, (chunk, at))?;
                if i + 1 == units {
                    step::head_of(gpu, w, body, head, chunk.len())?;
                }
            }
            body.held = pos + n;
            Ok(true)
        });
        // The call's own error first: an end refused after a failed call is its echo.
        let (gpu, _, body) = model.body_parts(CALL)?;
        let ended = body.stream_end(gpu.stream(), r.is_ok() && !history);
        let v = r?;
        ended?;
        Ok(v)
    })?;
    token.ok_or_else(|| GpuError::shape(CALL, "a prompt pass that read no token"))
}

/// One unit of `ids` at positions `at ..`: the embedding into the streams, then every layer's
/// two sub-layers over the unit's rows, whole or around the host tier's batch port where the
/// plan hosts experts, each layer's last rows tapped when armed.
fn unit(
    (gpu, w): (&Gpu, &Weights),
    body: &mut Body,
    (ids, at): (&[u32], u32),
) -> Result<(), GpuError> {
    let m = ids.len();
    body.parts.trunk.embed(gpu.stream(), ids)?;
    match body.placed.take() {
        Some(mut placed) => {
            let walked = crate::placed::walk_pass((gpu, w), body, &mut placed, (m, at));
            body.placed = Some(placed);
            walked?;
        }
        None => {
            for l in 0..body.sh.n_layer {
                step::attn(gpu, w, body, l, Run::Prompt(at, m))?;
                step::ffn_whole(gpu, w, body, l, m)?;
                step::close(gpu, w, body, l, m)?;
            }
        }
    }
    Ok(())
}
