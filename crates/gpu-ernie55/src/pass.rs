//! The prompt pass: a prompt runs as units of up to the load's pass rows, each unit through every
//! layer at once (each part's prompt form over the unit's rows, every row the step's at its
//! position), the head on the last row of the last unit. A load whose plan hosts experts has no
//! pass yet: the routed block's host share over a unit's rows (the common host tier's batch port,
//! V4.1's grouped host leg) is not built, so such a load is opened without it
//! ([`Body::open_placed`] refuses the pair by name) and steps its prompt.

use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError, GpuModel};

use crate::body::{Body, Run};
use crate::step;

/// What the pass's refusals name.
const CALL: &str = "ernie55::pass::prompt";

/// The prompt `tokens` from the model's position as the pass's units; the argmax after the last.
/// Refused by name: no id, a load with no pass, a call off the position the stores stand at. A
/// call that fails part-way leaves the stores where no step continues from: the next call
/// resets first.
pub fn prompt(model: &mut GpuModel<Body>, tokens: &[u32]) -> Result<u32, GpuError> {
    if tokens.is_empty() {
        return Err(GpuError::shape(CALL, "a prompt of no id"));
    }
    let n = u32::try_from(tokens.len()).map_err(|_| GpuError::shape(CALL, "ids past u32"))?;
    let token = model.run_rows(tokens.len(), CALL, |gpu, w, body, head, pos| {
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
                + u32::try_from(i * rows).map_err(|_| GpuError::shape(CALL, "a unit past u32"))?;
            unit((gpu, w), body, (chunk, at))?;
            if i + 1 == units {
                step::head_of(gpu, w, body, head, chunk.len())?;
            }
        }
        body.held = pos + n;
        Ok(true)
    })?;
    token.ok_or_else(|| GpuError::shape(CALL, "a prompt pass that read no token"))
}

/// One unit of `ids` at positions `at ..`: the embedding into the streams, then every layer's
/// two sub-layers over the unit's rows, each layer's last rows tapped when armed.
fn unit(
    (gpu, w): (&Gpu, &Weights),
    body: &mut Body,
    (ids, at): (&[u32], u32),
) -> Result<(), GpuError> {
    let m = ids.len();
    body.parts.trunk.embed(gpu.stream(), ids)?;
    for l in 0..body.sh.n_layer {
        step::attn(gpu, w, body, l, Run::Prompt(at, m))?;
        step::ffn_whole(gpu, w, body, l, m)?;
        step::close(gpu, w, body, l, m)?;
    }
    Ok(())
}
