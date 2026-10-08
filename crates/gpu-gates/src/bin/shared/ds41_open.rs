//! The V4.1 residency gates' shared open: the load through the binaries' open
//! ([`open`], and [`open_edited`] with an edit of its plan), printing nothing
//! ([`Quiet`]). `gate_ds41_callstream` and the residency clauses it runs
//! (`shared/ds41_residency.rs`) read it here.

use std::path::Path;

use app::arch::deepseek41::Ds41Cfg;
use app::{Loaded, OpenArgs, Session, SessionError};
use bloomery_gpu::model::StepMode;
use bloomery_gpu_deepseek41::body::{Body, OpenCfg};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::generate::Place;
use gguf::Split;
use model::placement::{Machine, Plan, workstation};

use crate::quiet::Quiet;

/// The target by `machine`, placement `place`'s, under `cfg` through the
/// binaries' open.
pub fn open(
    path: &Path,
    place: Place,
    machine: impl Fn(usize) -> Machine,
    cfg: &OpenCfg,
) -> Result<Session<Body>, GateError> {
    open_edited(path, place, machine, cfg, |_| Ok(()))
}

/// [`open`] with `edit` applied to the plan before the load
/// (`Loaded::open_edited`): a static load placed by a dumped card table
/// (`generate::place_table`).
pub fn open_edited(
    path: &Path,
    place: Place,
    machine: impl Fn(usize) -> Machine,
    cfg: &OpenCfg,
    edit: impl FnOnce(&mut Plan<'_>) -> Result<(), SessionError>,
) -> Result<Session<Body>, GateError> {
    let file = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let args = OpenArgs {
        place: place.name(),
        machine,
        ctx: usize::try_from(workstation::CTX_MAX)?,
        mode: StepMode::Graph,
        cfg: Ds41Cfg {
            open: cfg.clone(),
            feed: cfg.body.prefill,
            card_timing: false,
        },
    };
    let loaded = Loaded::<Body>::open_edited(file, args, &mut Quiet, |_, plan| edit(plan))?
        .ok_or("the open planned nothing")?;
    Ok(loaded.ready(&mut Quiet)?)
}
