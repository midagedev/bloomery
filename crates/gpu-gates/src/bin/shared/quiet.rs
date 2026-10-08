//! The records-free logs of a gate or binary that prints its own lines:
//! [`Quiet`] is the open's log ([`OpenLog`], for every body) and the verify
//! capture's ([`RowsLog`]), and answers each step with "go on".

use app::{Open, OpenLog, RowsLog, SessionError};
use bloomery_gpu::GpuModel;
use model::placement::{Machine, Plan};

/// Hears every step of an open and every verify capture, and records none.
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

    fn prompt_buffers(&mut self, _: &GpuModel<B>) -> Result<(), SessionError> {
        Ok(())
    }
}

impl RowsLog for Quiet {
    fn capture_rows(&mut self, _: usize, _: usize) -> Result<(), SessionError> {
        Ok(())
    }
}
