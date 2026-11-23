//! The Inkling 2 session: [`Body`] behind the session's traits — its prompt stepped one id at a
//! time ([`GpuModel::step`] over the slice: each id a decode step's bits, the head's argmax
//! after the last the call's) and its keep rule. The whole-card body opens by its own constructor
//! ([`GpuModel::open`]), so the session plans nothing. The body lives in the gpu crate's own
//! inkling2 arch, so it needs no feature.

use bloomery_gpu::arch::inkling2::Body;
use bloomery_gpu::{GpuError, GpuModel};

use crate::{Keep, Prompt, cut_where_it_stands, keep_where_it_stands};

const WHAT: &str = "inkling2 session";

impl Prompt for Body {
    /// The ids one step at a time ([`GpuModel::step`]), one readback after the last.
    fn prompt(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<u32, GpuError> {
        m.step(ids)
    }
}

impl Keep for Body {
    /// A windowed visit's ring holds the window it stands at alone: a cut keeps that position,
    /// or nothing.
    fn keepable(m: &GpuModel<Body>, n: u32) -> u32 {
        keep_where_it_stands(m, n)
    }

    /// Back to empty at 0 — a reset; any other position than the model's is refused by name.
    fn cut(m: &mut GpuModel<Body>, n: u32) -> Result<(), GpuError> {
        cut_where_it_stands(m, n, (WHAT, "the windowed rings take no position back"))
    }
}
