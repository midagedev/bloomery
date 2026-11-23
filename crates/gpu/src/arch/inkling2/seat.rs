//! Inkling 2's body behind the trait the common seats take positions back by: [`Rollback`], which
//! takes back the positions its conv ring and window rings still hold.

use super::body::Body;
use crate::GpuError;
use crate::model::Rollback;

const ROLLBACK: &str = "inkling2::Body::rollback";

impl Rollback for Body {
    /// Back to `pos`, no device work: every store is positional (a visit's K and V by position,
    /// its conv ring by `position mod rows`, the rest per token), and the next step or pass
    /// writes a position's rows before any launch reads them. The depth is measured from the
    /// written end ([`Body::rollback_floor`]): the conv ring holds [`Body::rollback_reach`]
    /// positions below it ([`runtime::stores::PASS_ROWS`]), and a position below that is refused
    /// by name. The written end stays, so a second rollback is as deep as the first and the span
    /// between.
    fn rollback(&mut self, pos: u32) -> Result<(), GpuError> {
        if self.held == u32::MAX {
            return Err(GpuError::shape(
                ROLLBACK,
                format!(
                    "a rollback to position {pos}: the stores stand nowhere (a call that failed \
                     part-way: reset)"
                ),
            ));
        }
        if pos > self.held {
            return Err(GpuError::shape(
                ROLLBACK,
                format!(
                    "a rollback to position {pos} past the {} the stores stand at",
                    self.held
                ),
            ));
        }
        let floor = self.rollback_floor();
        if pos < floor {
            return Err(GpuError::shape(
                ROLLBACK,
                format!(
                    "a rollback to position {pos} from {}: the conv ring holds {} positions \
                     below the written end, down to {floor}",
                    self.held,
                    self.rollback_reach()
                ),
            ));
        }
        self.held = pos;
        Ok(())
    }
}
