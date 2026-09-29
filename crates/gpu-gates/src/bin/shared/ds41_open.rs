//! The V4.1 residency gates' shared open: the records-free open log
//! ([`Quiet`]), the load through the binaries' open ([`open`]) and the
//! FNV-1a 64 of a logits row ([`fnv`]). `gate_deepseek41_residency` and
//! `gate_ds41_callstream` read them here.

use std::path::Path;

use app::arch::deepseek41::Ds41Cfg;
use app::{Loaded, OpenArgs, OpenLog, RowsLog, Session, SessionError};
use bloomery_gpu::model::StepMode;
use bloomery_gpu_deepseek41::body::{Body, Deepseek41Model, OpenCfg};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::generate::Place;
use gguf::Split;
use model::arch::deepseek41::place::PlanInputs;
use model::placement::{Machine, Plan, workstation};

/// The open's records: none; the gate prints its own lines.
pub struct Quiet;

impl OpenLog<Body> for Quiet {
    fn plan(
        &mut self,
        _: &'static str,
        _: &PlanInputs,
        _: &Machine,
        _: &Plan<'_>,
    ) -> Result<bool, SessionError> {
        Ok(true)
    }
    fn load(&mut self, _: &Deepseek41Model) -> Result<(), SessionError> {
        Ok(())
    }
    fn capture(&mut self, _: usize) -> Result<(), SessionError> {
        Ok(())
    }
    fn prompt_buffers(&mut self, _: &Deepseek41Model) -> Result<(), SessionError> {
        Ok(())
    }
}

impl RowsLog for Quiet {
    fn capture_rows(&mut self, _: usize, _: usize) -> Result<(), SessionError> {
        Ok(())
    }
}

/// The target by `machine` under `cfg` through the binaries' open.
pub fn open(
    path: &Path,
    machine: impl Fn(usize) -> Machine,
    cfg: &OpenCfg,
) -> Result<Session<Body>, GateError> {
    let file = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let args = OpenArgs {
        place: Place::Bp.name(),
        machine,
        ctx: usize::try_from(workstation::CTX_MAX)?,
        mode: StepMode::Graph,
        cfg: Ds41Cfg {
            open: cfg.clone(),
            feed: cfg.body.prefill,
            card_timing: false,
        },
    };
    let loaded = Loaded::<Body>::open(file, args, &mut Quiet)?.ok_or("the open planned nothing")?;
    Ok(loaded.ready(&mut Quiet)?)
}

pub fn fnv(row: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in row {
        for b in v.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}
