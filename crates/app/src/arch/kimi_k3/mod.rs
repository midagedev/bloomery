//! The Kimi K3 session: [`Body`] behind the session's traits. The body has no rows pass, so a
//! prompt is its ids stepped one at a time ([`GpuModel::step`] over the slice): each id is a
//! decode step's bits and the head's argmax after the last is the call's. A delta layer's state
//! holds the position it stands at alone, so a cut keeps every held position or nothing; the
//! latent layers' caches are per position and ask nothing of it. The body lives in the gpu crate's
//! own kimi_k3 arch, so it needs no feature.

use bloomery_gpu::arch::kimi_k3::Body;
use bloomery_gpu::{GpuError, GpuModel};

use crate::{Keep, Prompt};

const WHAT: &str = "kimi_k3 session";

impl Prompt for Body {
    /// The ids stepped from the model's position. Each step owns its residency pass
    /// ([`GpuModel::step`]: its boundary, its kept row, the next boundary ahead), and a host
    /// tier's open-call state refuses a boundary inside an open call, so no outer
    /// [`prompt_pass`](bloomery_gpu::host::served::prompt_pass) wraps the ids. The body's
    /// residency lever is off ([`Residency::Off`](bloomery_gpu::host::swap::Residency)), so no
    /// rule counts the prompt ids' decode rows.
    fn prompt(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<u32, GpuError> {
        m.step(ids)
    }
}

impl Keep for Body {
    /// The delta layers' state holds the position it stands at alone: a cut keeps that position,
    /// or nothing.
    fn keepable(m: &GpuModel<Body>, n: u32) -> u32 {
        if n >= m.pos() { m.pos() } else { 0 }
    }

    /// Back to empty at 0 (a reset), or the position the model stands at (nothing to take back);
    /// any other cut is refused by name, since the body takes no position back.
    fn cut(m: &mut GpuModel<Body>, n: u32) -> Result<(), GpuError> {
        match n {
            0 => m.reset(),
            n if n >= m.pos() => Ok(()),
            n => Err(GpuError::shape(
                WHAT,
                format!(
                    "a cut to position {n} of {}: the delta layers' state takes no position back",
                    m.pos()
                ),
            )),
        }
    }
}
