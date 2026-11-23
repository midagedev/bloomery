//! The Inkling 2 session: [`Body`] behind the session's traits — its prompt as the rows passes
//! ([`GpuModel::prompt_pass`]: the stores left where the ids' steps leave them, the head's argmax
//! after the last row the call's) and its keep rule (the positions its rollback takes back). The
//! whole-card body opens by its own constructor ([`GpuModel::open`]), so the session plans
//! nothing. The body lives in the gpu crate's own inkling2 arch, so it needs no feature.

use bloomery_gpu::arch::inkling2::Body;
use bloomery_gpu::{GpuError, GpuModel};

use crate::{Keep, Prompt, cut_positions, keep_down_to};

const WHAT: &str = "inkling2 session";

impl Prompt for Body {
    /// The ids as passes of rows ([`GpuModel::prompt_pass`]), one readback after the last.
    fn prompt(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<u32, GpuError> {
        m.prompt_pass(ids)
    }
}

impl Keep for Body {
    /// Every store is positional, so a cut keeps `n` down to [`Body::rollback_floor`], the written
    /// end less the conv ring's reach, and nothing below it ([`keep_down_to`]).
    fn keepable(m: &GpuModel<Body>, n: u32) -> u32 {
        match m.body(WHAT) {
            Ok(body) => keep_down_to(m, n, body.rollback_floor()),
            Err(_) => 0,
        }
    }

    /// Back to empty at 0 — a reset; else the model's rollback, which the body refuses by name
    /// below its floor.
    fn cut(m: &mut GpuModel<Body>, n: u32) -> Result<(), GpuError> {
        cut_positions(m, n)
    }
}
