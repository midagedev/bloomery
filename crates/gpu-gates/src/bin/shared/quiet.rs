//! The logs of a gate or binary that prints its own lines: [`Quiet`] is the
//! open's log ([`OpenLog`], for every body) and the verify capture's
//! ([`RowsLog`]), and answers each step with "go on". Its one record is the
//! card-byte invariant's ([`bloomery_gpu_gates::card_bytes::watch`]).

use app::{Open, OpenLog, RowsLog, SessionError};
use bloomery_gpu::GpuModel;
use model::placement::{Machine, Plan};

/// Hears every step of an open and every verify capture, and records only the
/// card-byte invariant's readings.
pub struct Quiet;

impl<B: Open> OpenLog<B> for Quiet {
    fn plan(
        &mut self,
        _: &'static str,
        _: &B::Inputs,
        _: &Machine,
        _: &Plan<'_>,
    ) -> Result<bool, SessionError> {
        Ok(true)
    }

    fn load(&mut self, _: &GpuModel<B>) -> Result<(), SessionError> {
        Ok(())
    }

    fn capture(&mut self, _: usize) -> Result<(), SessionError> {
        Ok(())
    }

    /// The card-byte invariant's readings, printed and refused as this process's
    /// card hold says ([`bloomery_gpu_gates::card_bytes::watch`]).
    fn card_watch(&mut self) -> Result<Box<dyn app::CardWatch>, SessionError> {
        bloomery_gpu_gates::card_bytes::watch()
    }
}

impl RowsLog for Quiet {
    fn capture_rows(&mut self, _: usize, _: usize, _: app::PhaseBytes) -> Result<(), SessionError> {
        Ok(())
    }
}
