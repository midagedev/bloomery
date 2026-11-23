//! Inkling 2's body behind the trait the common seats take positions back by: [`Rollback`], which
//! takes none back.

use super::body::Body;
use crate::GpuError;
use crate::model::Rollback;

const ROLLBACK: &str = "inkling2::Body::rollback";

impl Rollback for Body {
    /// Refused by name, always: a windowed visit's ring holds the window it stands at alone, so
    /// the body takes no position back. The session's cut keeps every held position or nothing
    /// (a reset) and never asks for another.
    fn rollback(&mut self, pos: u32) -> Result<(), GpuError> {
        Err(GpuError::shape(
            ROLLBACK,
            format!("a rollback to position {pos}: the windowed rings take no position back"),
        ))
    }
}
